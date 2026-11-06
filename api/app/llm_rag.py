"""LLM intelligence, phase 2: shaping what Chakudya's RAG routes return.

This module never calls the network and never stores anything. It takes the JSON that `llm_gateway.post_json` returned and produces the
response the laboratory gives its callers, applying the phase 1 rules to the *answer*:

- The answer is passed through `llm_policy.enforce_output`: an answer that states a diagnosis or a prescription about a person is
  replaced by the fixed refusal text, and its category is recorded (as a code, not as text).
- Citation markers ``[n]`` in the answer are checked against the sources Chakudya listed. A marker with no matching source is removed
  (never shown as if it were backed) and counted; an answer that cites nothing is flagged rather than presented as grounded.
- Intent is accepted only from Chakudya's published list; anything else becomes ``unknown``.
- Retrieved chunks are cut to a fixed length and only a few scalar metadata values are passed on, so a large or odd chunk cannot
  turn the endpoint into a bulk export of the knowledge base.

No new language model is introduced in this phase: the answer is whatever Chakudya's own pipeline produced.
"""
import re
from dataclasses import dataclass, field

from . import llm_policy

CONTEXTS = ("clinical", "general", "both")
INTENTS = ("food_search", "barcode_search", "nutrition_question", "exchange_list", "enteral_formula", "general_chat")
MAX_HISTORY = 6                  # Chakudya uses the last 6 turns
MAX_HISTORY_CHARS = 400          # ...and cuts each to 400 characters
MAX_SOURCES = 10
MAX_ANSWER_CHARS = 8000
MAX_SNIPPET_CHARS = 1200
MAX_TITLE_CHARS = 200

_MARKER = re.compile(r"\[(\d{1,3})\]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ShapeError(RuntimeError):
    """Chakudya answered with something that is not the documented shape. The message never contains response text."""


def _data(raw: dict) -> object:
    if raw.get("status") not in (None, "success"):
        raise ShapeError("Chakudya did not report success.")
    return raw.get("data")


def _text(value, limit: int) -> str:
    return _CONTROL.sub("", value)[:limit] if isinstance(value, str) else ""


def _sources(items) -> list[dict]:
    out = []
    if not isinstance(items, list):
        return out
    for n, it in enumerate(items[:MAX_SOURCES], start=1):
        if not isinstance(it, dict):
            continue
        sid = it.get("id")
        out.append({"id": sid if isinstance(sid, int) and not isinstance(sid, bool) and 0 < sid < 1000 else n,
                    "source": _text(it.get("source"), MAX_TITLE_CHARS), "title": _text(it.get("title"), MAX_TITLE_CHARS)})
    return out


@dataclass
class Shaped:
    body: dict
    audit: dict = field(default_factory=dict)      # counts and fixed codes only; safe for llm_audit.record


def shape_ask(raw: dict, *, context: str, top_k: int, history_turns: int, latency_ms: int) -> Shaped:
    d = _data(raw)
    if not isinstance(d, dict) or not isinstance(d.get("answer"), str):
        raise ShapeError("Chakudya returned an answer in an unexpected shape.")
    sources = _sources(d.get("sources"))
    ids = {s["id"] for s in sources}
    answer = _text(d["answer"], MAX_ANSWER_CHARS)

    found = [int(m) for m in _MARKER.findall(answer)]
    invalid = sorted({n for n in found if n not in ids})
    if invalid:
        answer = _MARKER.sub(lambda m: m.group(0) if int(m.group(1)) in ids else "", answer)
    cited = sorted({n for n in found if n in ids})

    shown, verdict = llm_policy.enforce_output(answer)
    blocked = not verdict.allowed
    if blocked:
        cited = []                                  # the refusal text cites nothing; do not show citations for withheld text
    intent = d.get("intent") if d.get("intent") in INTENTS else "unknown"
    cache = (raw.get("cache") or "").lower() if isinstance(raw.get("cache"), str) else ""
    warnings = []
    if blocked:
        warnings.append("The answer was withheld because it stated a diagnosis or a prescription about a person.")
    if invalid:
        warnings.append(f"{len(invalid)} citation marker(s) in the answer did not match any listed source and were removed.")
    if not blocked and sources and not cited:
        warnings.append("The answer does not cite any of the listed sources; treat it as ungrounded.")
    if not sources:
        warnings.append("No source supports this answer.")
    body = {
        "answer": shown, "intent": intent, "context": context,
        "grounded": bool(cited) and not blocked,
        "citations": [{**s, "cited": s["id"] in cited} for s in sources],
        "answer_withheld": blocked, "warnings": warnings, "latency_ms": latency_ms,
    }
    if blocked:
        body["withheld_categories"] = verdict.categories
    audit = {"context": context if context in CONTEXTS else "both", "top_k": top_k, "history_turns": history_turns, "intent": intent,
             "sources": len(sources), "cited": len(cited), "invalid_citations": len(invalid), "output_blocked": blocked,
             "latency_ms": latency_ms, "cache": cache if cache in ("hit", "miss") else "unknown"}
    return Shaped(body, audit)


def _meta(meta) -> dict:
    """A handful of short scalar values only; nested or long values are dropped."""
    if not isinstance(meta, dict):
        return {}
    out = {}
    for k, v in list(meta.items())[:8]:
        if isinstance(k, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", k) and isinstance(v, (str, int, float)) and not isinstance(v, bool):
            out[k] = _text(v, 120) if isinstance(v, str) else v
    return out


def shape_retrieve(raw: dict, *, context: str, top_k: int, latency_ms: int) -> Shaped:
    d = _data(raw)
    if not isinstance(d, list):
        raise ShapeError("Chakudya returned chunks in an unexpected shape.")
    chunks = []
    for n, c in enumerate(d[:top_k], start=1):
        if not isinstance(c, dict):
            continue
        content = c.get("content") if isinstance(c.get("content"), str) else ""
        score = next((c[k] for k in ("similarity", "score", "rerank_score") if isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool)), None)
        chunks.append({"rank": n, "source": _text(c.get("source"), MAX_TITLE_CHARS), "title": _text(c.get("title"), MAX_TITLE_CHARS),
                       "context": c.get("context") if c.get("context") in CONTEXTS else None,
                       "score": round(float(score), 4) if score is not None else None,
                       "snippet": _text(content, MAX_SNIPPET_CHARS), "truncated": len(content) > MAX_SNIPPET_CHARS,
                       "metadata": _meta(c.get("metadata"))})
    cache = (raw.get("cache") or "").lower() if isinstance(raw.get("cache"), str) else ""
    body = {"chunks": chunks, "count": len(chunks), "context": context, "latency_ms": latency_ms,
            "note": "Source text from the Chakudya knowledge base, not a generated answer. Snippets are shortened."}
    audit = {"context": context if context in CONTEXTS else "both", "top_k": top_k, "sources": len(chunks), "latency_ms": latency_ms,
             "cache": cache if cache in ("hit", "miss") else "unknown"}
    return Shaped(body, audit)
