"""Fine-tuning data (`/v1/llm/train`): reviewed question-and-answer examples, exported as JSONL for LoRA training on an external GPU.

- Examples are knowledge only, never patient text: every text passes the identifier detector and is rejected, never masked.
- The answer must cite the sources it was written from ([n] matching the passages) and must not state a diagnosis or prescription about a person.
- A question that matches a HELD-OUT gold question is refused, so the evaluation data can never leak into training.
- An example is exported only when approved by a different key than its author. The export uses the same prompt builder as comparison runs,
  so a model is trained on exactly the format it is asked at run time. The audit log holds counts only.
"""
import hashlib
import json
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import llm_audit, llm_eval, llm_openweight, llm_policy
from ..config import get_settings
from ..db import get_db
from ..models import LlmEvalQuestion, LlmTrainExample, utcnow
from ..security import Principal, require

router = APIRouter(prefix="/v1/llm/train", tags=["llm-train"])
MAX_PASSAGES, MAX_ANSWER = 6, 3000
_MARK = re.compile(r"\[(\d{1,3})\]")


class Example(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    passages: list[str] = Field(default_factory=list, max_length=MAX_PASSAGES)
    answer: str = Field(min_length=1, max_length=MAX_ANSWER)
    split: Literal["train", "val"] = "train"


class ExamplesCreate(BaseModel):
    examples: list[Example] = Field(min_length=1, max_length=50)


class ReviewBody(BaseModel):
    decision: Literal["approved", "rejected"]


def _norm(t: str) -> str:
    return " ".join(llm_policy.sanitize(t).casefold().split())


def _clean(e: Example, heldout: set[str], seen: set[str]) -> dict:
    q = " ".join(llm_policy.sanitize(e.question).split())
    ans = llm_policy.sanitize(e.answer).strip()
    passages = [" ".join(llm_policy.sanitize(p).split())[:llm_openweight.MAX_PASSAGE_CHARS] for p in e.passages if p.strip()]
    for field, text in [("question", q), ("answer", ans)] + [("passage", p) for p in passages]:
        llm_eval._no_identifiers(field, text)
    if _norm(q) in heldout:
        raise llm_eval.GoldError("question", "This question is in a held-out gold set. Held-out questions must never be used for training.")
    if _norm(q) in seen:
        raise llm_eval.GoldError("question", "This question already exists as a training example.")
    if not llm_policy.screen_output(ans).allowed:
        raise llm_eval.GoldError("answer", "The answer states a diagnosis or a prescription about a person. Rewrite it as knowledge.")
    marks = {int(m) for m in _MARK.findall(ans)}
    if passages and not marks:
        raise llm_eval.GoldError("answer", "Cite the sources the answer is written from, for example [1].")
    if any(m < 1 or m > len(passages) for m in marks):
        raise llm_eval.GoldError("answer", "A citation marker does not match any passage (passages are numbered from 1).")
    return {"question": q, "passages": passages, "answer": ans, "split": e.split}


def _out(x: LlmTrainExample) -> dict:
    return {"id": x.id, "question": x.question, "passages": [p["text"] for p in x.passages_json or []], "answer": x.answer, "split": x.split, "status": x.status,
            "created_by": x.created_by, "reviewed_by": x.reviewed_by, "created_at": x.created_at.isoformat() + "Z"}


@router.get("/examples", dependencies=[Depends(require("read:llm"))])
def list_examples(status: str | None = None, db: Session = Depends(get_db)):
    q = select(LlmTrainExample).order_by(LlmTrainExample.id.desc()).limit(300)
    if status:
        q = q.where(LlmTrainExample.status == status)
    counts = {s: n for s, n in db.execute(select(LlmTrainExample.status, func.count()).group_by(LlmTrainExample.status))}
    approved = {sp: n for sp, n in db.execute(select(LlmTrainExample.split, func.count()).where(LlmTrainExample.status == "approved").group_by(LlmTrainExample.split))}
    return {"counts": counts, "approved_by_split": approved, "examples": [_out(x) for x in db.scalars(q)],
            "note": "Only approved examples are exported. Approval needs a different key than the author."}


@router.post("/examples", status_code=201)
def add_examples(body: ExamplesCreate, request: Request, p: Principal = Depends(require("write:llm_gold")), db: Session = Depends(get_db)):
    heldout = {_norm(q.question) for q in db.scalars(select(LlmEvalQuestion).where(LlmEvalQuestion.split == "heldout"))}
    seen = {_norm(x) for x in db.scalars(select(LlmTrainExample.question))}
    clean = []
    for i, e in enumerate(body.examples, 1):
        try:
            c = _clean(e, heldout, seen)
        except llm_eval.GoldError as err:
            raise HTTPException(422, f"Example {i}: {err}")
        seen.add(_norm(c["question"]))
        clean.append(c)
    rows = [LlmTrainExample(question=c["question"], passages_json=[{"text": t} for t in c["passages"]], answer=c["answer"], split=c["split"],
                            created_by=p.name, created_by_key_id=p.key_id) for c in clean]
    db.add_all(rows)
    db.commit()
    llm_audit.record(db, "llm.train.examples_added", principal=p, request=request, examples=len(rows))
    return {"added": len(rows), "examples": [_out(r) for r in rows]}


@router.post("/examples/{eid}/review")
def review(eid: int, body: ReviewBody, request: Request, p: Principal = Depends(require("rate:llm")), db: Session = Depends(get_db)):
    x = db.get(LlmTrainExample, eid)
    if x is None:
        raise HTTPException(404, "Unknown example.")
    if get_settings().require_separate_reviewer and x.created_by_key_id == p.key_id:
        raise HTTPException(403, "A different person than the author must review an example.")
    x.status, x.reviewed_by, x.reviewed_by_key_id, x.reviewed_at = body.decision, p.name, p.key_id, utcnow()
    db.commit()
    llm_audit.record(db, "llm.train.reviewed", principal=p, request=request, resource_id=eid, review=body.decision, examples=1)
    return _out(x)


@router.delete("/examples/{eid}")
def delete_example(eid: int, request: Request, p: Principal = Depends(require("write:llm_gold")), db: Session = Depends(get_db)):
    x = db.get(LlmTrainExample, eid)
    if x is None:
        raise HTTPException(404, "Unknown example.")
    if x.status == "approved":
        raise HTTPException(409, "An approved example cannot be deleted. Reject it instead.")
    if x.created_by_key_id != p.key_id and p.role != "admin":
        raise HTTPException(403, "Only the author or an administrator can delete a draft.")
    db.delete(x)
    db.commit()
    llm_audit.record(db, "llm.train.deleted", principal=p, request=request, resource_id=eid, examples=1)
    return {"deleted": eid}


@router.get("/export")
def export(request: Request, split: Literal["train", "val"] = "train", p: Principal = Depends(require("export:llm_train")), db: Session = Depends(get_db)):
    """JSONL of approved examples, one chat per line, in the format `training/train_lora.py` reads."""
    rows = list(db.scalars(select(LlmTrainExample).where(LlmTrainExample.status == "approved", LlmTrainExample.split == split).order_by(LlmTrainExample.id)))
    if not rows:
        raise HTTPException(404, "There are no approved examples for that split.")
    lines = []
    for x in rows:
        msgs = llm_openweight.build_rag_messages(x.question, x.passages_json or []) + [{"role": "assistant", "content": x.answer}]
        lines.append(json.dumps({"messages": msgs}, ensure_ascii=False))
    data = ("\n".join(lines) + "\n").encode()
    sha = hashlib.sha256(data).hexdigest()
    llm_audit.record(db, "llm.train.exported", principal=p, request=request, examples=len(rows), split=split)
    return Response(data, media_type="application/x-ndjson",
                    headers={"Content-Disposition": f'attachment; filename="cnr-{split}.jsonl"', "X-Content-SHA256": sha, "X-Examples": str(len(rows)),
                             "Access-Control-Expose-Headers": "X-Content-SHA256, X-Examples"})
