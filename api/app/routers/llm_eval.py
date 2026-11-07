"""LLM intelligence, phase 3: the RAG evaluation laboratory (`/v1/llm/eval/...`).

Dietitians write a gold-standard question set (expected sources and facts, or "should refuse"). A run sends every active question through
the same gate as a live question and the server-side gateway to Chakudya's /rag/retrieve and /rag/ask, and records retrieval hit,
citation validity, groundedness, refusals and latency per question. Reviewers then rate each answer. A summary states the results with
confidence intervals and says plainly whether there is enough evidence to make any quality claim (`readiness`).

Rules carried over from phase 1: gold questions must pass the identifier detector (no patient information); every call to Chakudya
counts against the starting key's rate limit; the audit log holds counts and fixed codes only (never questions, answers or comments).
Stored: gold questions and expected facts (written by dietitians), per-question metrics, the screened answer text (identifiers masked, so
reviewers can rate it) and ratings. No new language model is introduced.
"""
import logging
import time
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import llm_audit, llm_eval, llm_gateway, llm_openweight, llm_policy, nlp_lab
from ..config import get_settings
from ..db import SessionLocal, get_db
from ..models import LlmEvalQuestion, LlmEvalRating, LlmEvalResult, LlmEvalRun, LlmEvalSet, utcnow
from ..security import Principal, require

log = logging.getLogger("cnr-ml")
router = APIRouter(prefix="/v1/llm/eval", tags=["llm-eval"])
_sleep = time.sleep           # replaced in tests
STALE_RUN_SECONDS = 3 * 3600  # a run older than this is treated as dead (for example, lost in a restart)


# ---------------------------------------------------------------- shapes
class GoldQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    context: Literal["clinical", "general", "both"] = "both"
    expected_sources: list[str] = Field(default_factory=list, max_length=llm_eval.MAX_LIST_ITEMS)
    expected_facts: list[str] = Field(default_factory=list, max_length=llm_eval.MAX_LIST_ITEMS)
    should_refuse: bool = False
    split: Literal["dev", "heldout"] = "dev"


class SetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)


class QuestionsCreate(BaseModel):
    questions: list[GoldQuestion] = Field(min_length=1, max_length=50)


class RunCreate(BaseModel):
    split: Literal["dev", "heldout", "all"] = "dev"
    top_k: int = Field(default=5, ge=1, le=10)
    model: str | None = Field(default=None, max_length=60)    # an open-weight model id: answers are written by it from the same retrieved sources


class RatingBody(BaseModel):
    accuracy: int = Field(ge=1, le=5)
    supported: bool
    safe: bool
    comment: str = Field(default="", max_length=2000)


def _q_dict(q: LlmEvalQuestion) -> dict:
    return {"id": q.id, "question": q.question, "context": q.context, "expected_sources": list(q.expected_sources_json or []),
            "expected_facts": list(q.expected_facts_json or []), "should_refuse": q.should_refuse, "split": q.split}


def _q_out(q: LlmEvalQuestion) -> dict:
    return {**_q_dict(q), "archived": q.archived_at is not None, "created_by": q.created_by, "created_at": q.created_at.isoformat() + "Z"}


def _active(db: Session, set_id: int) -> list[LlmEvalQuestion]:
    return list(db.scalars(select(LlmEvalQuestion).where(LlmEvalQuestion.set_id == set_id, LlmEvalQuestion.archived_at.is_(None)).order_by(LlmEvalQuestion.id)))


def _set(db: Session, set_id: int) -> LlmEvalSet:
    row = db.get(LlmEvalSet, set_id)
    if row is None:
        raise HTTPException(404, "Unknown question set.")
    return row


def _bad(e: llm_eval.GoldError, prefix: str = "") -> HTTPException:
    return HTTPException(422, prefix + str(e))


def _no_identifiers(field: str, text: str) -> None:
    llm_eval._no_identifiers(field, text)


# ---------------------------------------------------------------- method
@router.get("/method", dependencies=[Depends(require("read:llm"))])
def method():
    return {**llm_eval.METHOD, "limits_per_set": {"max_questions": llm_eval.MAX_QUESTIONS_PER_SET, "max_question_chars": llm_eval.MAX_QUESTION_CHARS},
            "stored": "Gold questions and expected facts, per-question metrics, the screened answer text (identifiers masked) and reviewer ratings. "
                      "The audit log holds counts and fixed codes only."}


# ---------------------------------------------------------------- sets and questions
@router.get("/sets", dependencies=[Depends(require("read:llm"))])
def list_sets(db: Session = Depends(get_db)):
    out = []
    for s in db.scalars(select(LlmEvalSet).order_by(LlmEvalSet.id.desc())):
        qs = _active(db, s.id)
        runs = db.scalar(select(func.count()).select_from(LlmEvalRun).where(LlmEvalRun.set_id == s.id)) or 0
        out.append({"id": s.id, "name": s.name, "description": s.description, "created_by": s.created_by, "created_at": s.created_at.isoformat() + "Z",
                    "questions": len(qs), "answerable": sum(1 for q in qs if not q.should_refuse), "should_refuse": sum(1 for q in qs if q.should_refuse),
                    "heldout": sum(1 for q in qs if q.split == "heldout"), "runs": runs})
    return out


@router.post("/sets", status_code=201)
def create_set(body: SetCreate, request: Request, p: Principal = Depends(require("write:llm_gold")), db: Session = Depends(get_db)):
    name = " ".join(body.name.split())
    try:
        _no_identifiers("name", name)
        if body.description.strip():
            _no_identifiers("description", body.description)
    except llm_eval.GoldError as e:
        raise _bad(e)
    row = LlmEvalSet(name=name, description=body.description.strip(), created_by=p.name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "A question set with this name already exists.")
    llm_audit.record(db, "llm.eval.set_created", principal=p, request=request, resource_id=row.id, name_chars=len(name))
    return {"id": row.id, "name": row.name, "description": row.description}


@router.get("/sets/{set_id}", dependencies=[Depends(require("read:llm"))])
def get_set(set_id: int, db: Session = Depends(get_db)):
    s = _set(db, set_id)
    rows = list(db.scalars(select(LlmEvalQuestion).where(LlmEvalQuestion.set_id == set_id).order_by(LlmEvalQuestion.id)))
    active = [_q_dict(q) for q in rows if q.archived_at is None]
    return {"id": s.id, "name": s.name, "description": s.description, "created_by": s.created_by, "created_at": s.created_at.isoformat() + "Z",
            "sha256": llm_eval.set_sha256(active), "limits": {"max_questions": llm_eval.MAX_QUESTIONS_PER_SET},
            "questions": [_q_out(q) for q in rows]}


@router.post("/sets/{set_id}/questions", status_code=201)
def add_questions(set_id: int, body: QuestionsCreate, request: Request, p: Principal = Depends(require("write:llm_gold")), db: Session = Depends(get_db)):
    _set(db, set_id)
    existing = _active(db, set_id)
    if len(existing) + len(body.questions) > llm_eval.MAX_QUESTIONS_PER_SET:
        raise HTTPException(422, f"A set holds at most {llm_eval.MAX_QUESTIONS_PER_SET} active questions.")
    seen = {" ".join(q.question.casefold().split()) for q in existing}
    clean = []
    for i, raw in enumerate(body.questions, start=1):
        try:
            c = llm_eval.check_gold(raw.model_dump())
        except llm_eval.GoldError as e:
            raise _bad(e, f"Question {i}: ")
        key = " ".join(c["question"].casefold().split())
        if key in seen:
            raise HTTPException(422, f"Question {i}: this question is already in the set.")
        seen.add(key)
        clean.append(c)
    rows = [LlmEvalQuestion(set_id=set_id, question=c["question"], context=c["context"], expected_sources_json=c["expected_sources"],
                            expected_facts_json=c["expected_facts"], should_refuse=c["should_refuse"], split=c["split"], created_by=p.name) for c in clean]
    db.add_all(rows)
    db.commit()
    llm_audit.record(db, "llm.eval.questions_added", principal=p, request=request, resource_id=set_id, questions=len(rows),
                     expected_sources=sum(len(c["expected_sources"]) for c in clean), expected_facts=sum(len(c["expected_facts"]) for c in clean))
    return {"added": len(rows), "questions": [_q_out(r) for r in rows]}


@router.post("/questions/{qid}/archive")
def archive_question(qid: int, request: Request, p: Principal = Depends(require("write:llm_gold")), db: Session = Depends(get_db)):
    q = db.get(LlmEvalQuestion, qid)
    if q is None:
        raise HTTPException(404, "Unknown question.")
    if q.archived_at is None:
        q.archived_at = utcnow()
        db.commit()
        llm_audit.record(db, "llm.eval.question_archived", principal=p, request=request, resource_id=qid, questions=1)
    return _q_out(q)


# ---------------------------------------------------------------- runs
def _run_out(db: Session, r: LlmEvalRun) -> dict:
    done = db.scalar(select(func.count()).select_from(LlmEvalResult).where(LlmEvalResult.run_id == r.id)) or 0
    s = db.get(LlmEvalSet, r.set_id)
    cfg = {k: v for k, v in (r.config_json or {}).items() if k != "question_ids"}
    return {"id": r.id, "set_id": r.set_id, "set_name": s.name if s else None, "set_sha256": r.set_sha256, "status": r.status, "stop_reason": r.stop_reason,
            "question_count": r.question_count, "completed": done, "config": cfg, "created_by": r.created_by,
            "created_at": r.created_at.isoformat() + "Z", "finished_at": r.finished_at.isoformat() + "Z" if r.finished_at else None}


def _run(db: Session, run_id: int) -> LlmEvalRun:
    r = db.get(LlmEvalRun, run_id)
    if r is None:
        raise HTTPException(404, "Unknown evaluation run.")
    return r


def _rows(db: Session, run_id: int) -> tuple[list[dict], dict[int, list[dict]]]:
    results = list(db.scalars(select(LlmEvalResult).where(LlmEvalResult.run_id == run_id).order_by(LlmEvalResult.id)))
    qs = {q.id: q for q in db.scalars(select(LlmEvalQuestion).where(LlmEvalQuestion.id.in_([x.question_id for x in results] or [0])))}
    ratings: dict[int, list[dict]] = {}
    for g in db.scalars(select(LlmEvalRating).where(LlmEvalRating.result_id.in_([x.id for x in results] or [0]))):
        ratings.setdefault(g.result_id, []).append({"accuracy": g.accuracy, "supported": g.supported, "safe": g.safe, "reviewer": g.reviewer_key_id})
    rows = [{"result_id": x.id, "metrics": x.metrics_json or {}, "question": _q_dict(qs[x.question_id])} for x in results if x.question_id in qs]
    return rows, ratings


@router.get("/runs", dependencies=[Depends(require("read:llm"))])
def list_runs(set_id: int | None = None, db: Session = Depends(get_db)):
    q = select(LlmEvalRun).order_by(LlmEvalRun.id.desc()).limit(100)
    if set_id is not None:
        q = q.where(LlmEvalRun.set_id == set_id)
    return [_run_out(db, r) for r in db.scalars(q)]


@router.post("/sets/{set_id}/runs", status_code=202)
def start_run(set_id: int, body: RunCreate, request: Request, background: BackgroundTasks, p: Principal = Depends(require("use:llm")), db: Session = Depends(get_db)):
    s = get_settings()
    _set(db, set_id)
    bad = llm_gateway.config_problem(s)
    if bad:
        raise HTTPException(503, bad)
    qs = [q for q in _active(db, set_id) if body.split == "all" or q.split == body.split]
    if not qs:
        raise HTTPException(422, "There are no active questions in this set for that split.")
    now = utcnow()
    for r in db.scalars(select(LlmEvalRun).where(LlmEvalRun.status.in_(["queued", "running"]))):
        if (now - r.created_at).total_seconds() < STALE_RUN_SECONDS:
            raise HTTPException(409, "Another evaluation run is still running. Wait for it to finish or stop it.")
    dicts = [_q_dict(q) for q in qs]
    host = llm_gateway.public_status(s).get("chakudya_host")
    if body.model:
        if body.model not in llm_openweight.models(s):
            raise HTTPException(422, "That model is not in the allow-list. Choose one of: " + ", ".join(llm_openweight.models(s)) + ".")
        bad = llm_openweight.problem(s)
        if bad:
            raise HTTPException(503, bad)
    cfg = {"split": body.split, "top_k": body.top_k, "question_ids": [q["id"] for q in dicts], "identifier_policy": "reject", "policy": llm_policy.POLICY_VERSION,
           "identifier_detector": nlp_lab.ENGINE_VERSION, "method": llm_eval.METHOD["version"], "chakudya_host": host,
           "service_version": s.service_version,
           "language_model": "Chakudya's own pipeline (none added by this laboratory)" if not body.model else "open-weight model " + body.model + " (answers written from Chakudya's retrieved sources)",
           "generator": body.model or "chakudya", "prompt": "cnr-rag-prompt/1" if body.model else None}
    run = LlmEvalRun(set_id=set_id, set_sha256=llm_eval.set_sha256(dicts), config_json=cfg, status="queued", question_count=len(dicts),
                     created_by=p.name, created_by_key_id=p.key_id)
    db.add(run)
    db.commit()
    llm_audit.record(db, "llm.eval.run_started", principal=p, request=request, resource_id=run.id, questions=len(dicts), split=body.split, top_k=body.top_k,
                     **({"model": body.model, "provider": "openweight"} if body.model else {}))
    background.add_task(run_eval, run.id)
    return _run_out(db, run)


def run_eval(run_id: int) -> None:
    """Runs in a worker thread after the response is sent, with its own database session. Results are saved one question at a time, so a
    stopped or interrupted run keeps what it finished."""
    s = get_settings()
    with SessionLocal() as db:
        run = db.get(LlmEvalRun, run_id)
        if run is None:
            return
        run.status = "running"
        db.commit()
        cfg, status, reason, done, errors = run.config_json, "completed", None, 0, 0
        gen = None
        if cfg.get("generator") not in (None, "chakudya"):
            mdl = cfg["generator"]

            def gen(question, passages):
                try:
                    return llm_openweight.chat_text(s, mdl, llm_openweight.build_rag_messages(question, passages), max_tokens=600)["content"]
                except llm_openweight.OpenWeightError:
                    raise llm_eval.GeneratorError()
        try:
            for qid in cfg["question_ids"]:
                db.refresh(run)
                if run.cancel_requested:
                    status, reason = "stopped", "cancelled"
                    break
                row = db.get(LlmEvalQuestion, qid)
                try:
                    metrics, answer = llm_eval.evaluate_question(s, _q_dict(row), top_k=cfg["top_k"], key_id=run.created_by_key_id, sleep=_sleep, generator=gen)
                except llm_eval.RunStop as e:
                    status, reason = "stopped", e.reason
                    break
                db.add(LlmEvalResult(run_id=run.id, question_id=qid, metrics_json=metrics, answer_text=answer))
                db.commit()
                done += 1
                errors += 1 if metrics.get("outcome") == "error" else 0
        except Exception:
            log.exception("LLM evaluation run %s failed", run_id)
            db.rollback()
            status, reason = "failed", "unexpected_error"
        run.status, run.stop_reason, run.finished_at = status, reason, utcnow()
        db.commit()
        who = Principal(key_id=run.created_by_key_id or 0, name=run.created_by or "", role="background")
        llm_audit.record(db, "llm.eval.run_finished", principal=who, resource_id=run.id, questions=run.question_count, completed=done, errors=errors,
                         **({"reason": reason} if reason and reason.replace("_", "").isalnum() else {}))


@router.get("/runs/{run_id}", dependencies=[Depends(require("read:llm"))])
def get_run(run_id: int, db: Session = Depends(get_db)):
    r = _run(db, run_id)
    rows, ratings = _rows(db, run_id)
    out = _run_out(db, r)
    out["summary"] = llm_eval.summarize(rows, ratings, run_status=r.status, question_total=r.question_count)
    out["results"] = [{"result_id": x["result_id"], "question_id": x["question"]["id"], "question": x["question"]["question"], "context": x["question"]["context"],
                       "should_refuse": x["question"]["should_refuse"], "split": x["question"]["split"], "outcome": x["metrics"].get("outcome"),
                       "retrieval_hit": x["metrics"].get("retrieval_hit"), "cited_source_hit": x["metrics"].get("cited_source_hit"),
                       "grounded": x["metrics"].get("grounded"), "numbers_unsupported": x["metrics"].get("numbers_unsupported"),
                       "facts_found": x["metrics"].get("facts_found"), "facts_total": x["metrics"].get("facts_total"),
                       "latency_ask_ms": x["metrics"].get("latency_ask_ms"), "ratings": len(ratings.get(x["result_id"], []))} for x in rows]
    out["method"] = llm_eval.METHOD["version"]
    return out


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: int, request: Request, p: Principal = Depends(require("use:llm")), db: Session = Depends(get_db)):
    r = _run(db, run_id)
    if r.created_by_key_id != p.key_id and p.role != "admin":
        raise HTTPException(403, "Only the person who started a run, or an administrator, can stop it.")
    if r.status not in ("queued", "running"):
        raise HTTPException(409, "This run is not running.")
    r.cancel_requested = True
    db.commit()
    llm_audit.record(db, "llm.eval.run_stop_requested", principal=p, request=request, resource_id=run_id)
    return _run_out(db, r)


# ---------------------------------------------------------------- compare two runs
@router.get("/compare", dependencies=[Depends(require("read:llm"))])
def compare_runs(a: int, b: int, db: Session = Depends(get_db)):
    """Run A (for example the base model) against run B (for example a fine-tuned adapter). Only runs on the same question fingerprint can be compared."""
    ra, rb = _run(db, a), _run(db, b)
    if a == b:
        raise HTTPException(422, "Choose two different runs.")
    if ra.set_sha256 != rb.set_sha256:
        raise HTTPException(409, "These runs used different questions (the fingerprints differ), so they cannot be compared. Run both on the same set and split.")
    out = []
    for r in (ra, rb):
        rows, ratings = _rows(db, r.id)
        out.append({"label": f"#{r.id} " + ((r.config_json or {}).get("generator") or "chakudya"), "rows": rows, "ratings": ratings, "status": r.status, "total": r.question_count})
    res = llm_eval.compare(out[0], out[1])
    res["a"].update(run_id=a, generator=(ra.config_json or {}).get("generator") or "chakudya")
    res["b"].update(run_id=b, generator=(rb.config_json or {}).get("generator") or "chakudya")
    return res


# ---------------------------------------------------------------- results and reviewer ratings
@router.get("/results/{result_id}")
def get_result(result_id: int, p: Principal = Depends(require("read:llm")), db: Session = Depends(get_db)):
    x = db.get(LlmEvalResult, result_id)
    if x is None:
        raise HTTPException(404, "Unknown result.")
    q = db.get(LlmEvalQuestion, x.question_id)
    rs = list(db.scalars(select(LlmEvalRating).where(LlmEvalRating.result_id == x.id).order_by(LlmEvalRating.id)))
    return {"result_id": x.id, "run_id": x.run_id, "question": _q_out(q), "metrics": x.metrics_json, "answer": x.answer_text,
            "answer_note": "The answer as the laboratory returned it, after the diagnosis and prescription screen, with identifiers masked. Empty when the question was refused or withheld.",
            "ratings": [{"reviewer": g.reviewer_name, "accuracy": g.accuracy, "supported": g.supported, "safe": g.safe, "comment": g.comment,
                         "updated_at": g.updated_at.isoformat() + "Z", "mine": g.reviewer_key_id == p.key_id} for g in rs],
            "rating_scale": {"accuracy": "1 = wrong or misleading, 3 = partly correct or incomplete, 5 = fully correct",
                             "supported": "Every claim is backed by the cited sources", "safe": "No diagnosis, prescription or harmful advice"}}


@router.post("/results/{result_id}/rating")
def rate_result(result_id: int, body: RatingBody, request: Request, p: Principal = Depends(require("rate:llm")), db: Session = Depends(get_db)):
    x = db.get(LlmEvalResult, result_id)
    if x is None:
        raise HTTPException(404, "Unknown result.")
    if (x.metrics_json or {}).get("outcome") == "error":
        raise HTTPException(409, "A question that ended in an error has no answer to rate.")
    try:
        comment = llm_eval.check_comment(body.comment)
    except llm_eval.GoldError as e:
        raise _bad(e)
    g = db.scalar(select(LlmEvalRating).where(LlmEvalRating.result_id == result_id, LlmEvalRating.reviewer_key_id == p.key_id))
    if g is None:
        g = LlmEvalRating(result_id=result_id, reviewer_key_id=p.key_id, reviewer_name=p.name, accuracy=body.accuracy, supported=body.supported, safe=body.safe, comment=comment)
        db.add(g)
    else:
        g.accuracy, g.supported, g.safe, g.comment, g.updated_at = body.accuracy, body.supported, body.safe, comment, utcnow()
    db.commit()
    llm_audit.record(db, "llm.eval.rated", principal=p, request=request, resource_id=result_id, accuracy=body.accuracy, supported=body.supported,
                     safe=body.safe, comment_chars=len(comment))
    return {"result_id": result_id, "accuracy": g.accuracy, "supported": g.supported, "safe": g.safe, "comment": g.comment}
