"""LLM intelligence, phase 4: summarising a de-identified clinical note, restricted to facts the rule engine extracted.

Pipeline (the endpoint in routers/llm_summary.py runs it in this order):
 1. identifier gate on the note (`llm_policy.prepare_for_external`, kind "document"): identifiable text is refused (default) or masked;
 2. fact extraction by the rule engine (`llm_extract`): the facts, each with the exact span of the note it came from;
 3. the model (Groq-hosted, open weights) is shown ONLY the facts (`llm_extract.model_view`), never the note, and asked for numbered
    sentences that each cite the fact ids they use;
 4. every sentence is validated against the facts it cites and against the source (`validate_output`); a sentence that fails is dropped,
    not repaired, and is not returned;
 5. the result is "pending clinician review": nothing is stored, and the review is an attestation recorded as counts only.

What validation proves, and what it does not. A sentence that passes: contains no number that is not in the note AND in a cited fact; its
measurements (as read back by the same rule engine) match a cited fact in type, value and unit; its condition, symptom, food, intervention,
nutrient and population terms are cited facts with the same negated / uncertain / historical / family status; it holds no identifier and
states no diagnosis or prescription about a person. It does NOT prove that the sentence is complete, well worded or clinically right, that
a word outside the lexicon is true (such words are shown to the reviewer as `unverified_words`), or that the extraction itself was right.
That is why clinician review is required.
"""
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

from . import llm_extract, llm_policy, nlp_lab

PROMPT_VERSION = "cnr-summary-prompt/1"
PROMPT_CODE = "cnr-summary-prompt-1"
MAX_SENTENCES = 8
MAX_SENTENCE_CHARS = 400
MAX_FACTS_PER_SENTENCE = 8

SUMMARY_RULES = (
    "\nYou are writing a short summary of a clinical note for a clinician. You are given a numbered list of FACTS as JSON. Use only those facts.\n"
    "Answer with one JSON object and nothing else: {\"sentences\": [{\"text\": \"...\", \"facts\": [\"F1\", \"F2\"]}]}\n"
    "6. Write 1 to 8 short sentences. Every sentence must list, in \"facts\", the ids of the facts it uses, and use only those.\n"
    "7. Copy numbers and units exactly as given. Never round, convert, add, subtract or estimate a number. Never write a number that is not in a fact.\n"
    "8. Mention nothing that is not in a fact: no causes, no explanations, no diagnosis, no interpretation, no advice, no doses, no plans.\n"
    "9. Keep each fact's status. status \"negated\": write it as \"No <term> documented\". status \"uncertain\": write \"Suspected <term>\". "
    "status \"historical\": write \"History of <term>\". status \"family\": write \"Family history of <term>\". status \"present\": state the term plainly.\n"
    "10. Describe what the note records, for example \"The note records ...\". Do not say that a person has, or probably has, a condition.\n"
    "11. If the facts are too few to summarise, answer {\"sentences\": []}.\n")
SYSTEM_PROMPT = llm_policy.SYSTEM_RULES + SUMMARY_RULES

REVIEW_CHECKLIST = [
    "I compared every fact (with its source span) against the note.",
    "I compared every number and unit in the summary against the note.",
    "I understand this is decision support only: I remain responsible for anything entered in a record or used for a patient.",
]

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")
_NUMWORD = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                      r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|half|twice|dozen|first|second|third)\b", re.I)
# Ordinary words a summary needs that are not clinical terms; anything else must appear in the note or in a fact, or it is shown to the reviewer.
_CONNECTORS = set("""note notes records recorded record documented documents document states stated reports reported report with without that this
the and for from were was has have had are also noted history family suspected measured measurement measurements weight length height
aged age months month years year days day weeks week patient child infant mother foods food eating given started current present absent
level levels value values mentioned described findings finding""".split())


# ---------------------------------------------------------------- prompt
def build_messages(facts: list[dict]) -> list[dict]:
    """The only things sent to the model: the fixed rules and the facts (lexicon terms, numbers, units, statuses). Never the note."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "FACTS:\n" + json.dumps(llm_extract.model_view(facts), ensure_ascii=False)}]


# ---------------------------------------------------------------- validation
class BadOutput(RuntimeError):
    """The model did not answer in the requested JSON shape. The message never contains model text."""


@dataclass
class Sentence:
    text: str
    facts: list[str]
    flags: list[dict] = field(default_factory=list)


@dataclass
class Validation:
    accepted: list[Sentence]
    reasons: Counter            # UPPER_CASE reason code -> count of rejected sentences
    numbers_checked: int
    rejected: int

    @property
    def audit_reasons(self) -> dict:
        return dict(self.reasons)


def parse_output(content: str) -> list[dict]:
    try:
        d = json.loads(content)
    except ValueError:
        raise BadOutput("The model did not answer with JSON.")
    items = d.get("sentences") if isinstance(d, dict) else None
    if not isinstance(items, list):
        raise BadOutput("The model's JSON did not have the requested shape.")
    out = []
    for it in items[:MAX_SENTENCES]:
        if isinstance(it, dict):
            out.append({"text": it.get("text"), "facts": it.get("facts")})
        else:
            out.append({"text": None, "facts": None})
    return out


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "")}


def _stem(w: str) -> str:
    return w[:-2] if w.endswith("es") and len(w) > 5 else w[:-1] if w.endswith("s") and len(w) > 4 else w


def _fact_numbers(f: dict) -> set[float]:
    n = {float(f["value"])} if f.get("value") is not None else set()
    return n | llm_extract.numbers(f.get("evidence", ""))


def _same(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) < 1e-9


def validate_sentence(item: dict, facts_by_id: dict[str, dict], src_numbers: set[float], src_words: set[str], vocab: set[str]) -> tuple[Sentence | None, list[str], int]:
    """(sentence or None, reason codes, numbers checked). A sentence is returned only when no reason code was raised."""
    reasons: list[str] = []
    text, ids = item.get("text"), item.get("facts")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SENTENCE_CHARS:
        return None, ["FORMAT"], 0
    text = " ".join(llm_policy.sanitize(text).split())
    if not isinstance(ids, list) or not ids or len(ids) > MAX_FACTS_PER_SENTENCE or not all(isinstance(i, str) and i in facts_by_id for i in ids):
        return None, ["NO_FACT"], 0
    cited = [facts_by_id[i] for i in dict.fromkeys(ids)]

    if nlp_lab.screen_identifiers(text):
        reasons.append("IDENTIFIER")
    if not llm_policy.screen_output(text).allowed:
        reasons.append("SCREEN")

    # numbers: every number must be in the note AND in a cited fact; spelled-out numbers must be in the note
    allowed = set().union(*(_fact_numbers(f) for f in cited))
    nums = llm_extract.numbers(text)
    if any(n not in src_numbers or n not in allowed for n in nums):
        reasons.append("NUMBER")
    if any(m.lower() not in src_words for m in _NUMWORD.findall(text)):
        reasons.append("NUMBER")

    # entities and measurements: read the sentence back with the same rule engine and compare with the cited facts
    a = nlp_lab.analyze(text)
    for e in a["entities"]:
        if e["label"] == "PHI":
            continue
        if e["label"] == "MEASUREMENT":
            ok = any(f["kind"] == "measurement" and f.get("mtype") == e.get("mtype") and _same(f.get("value"), e.get("value"))
                     and (f.get("unit") or "").lower() == (e.get("unit") or "").lower() for f in cited)
            if not ok:
                reasons.append("MEASUREMENT")
            continue
        kind = llm_extract._KIND.get(e["label"])
        same_term = [f for f in cited if f["kind"] == kind and f["term"] == e["norm"]]
        if not same_term:
            reasons.append("ENTITY")
        elif llm_extract._status(e.get("attributes")) not in {f["status"] for f in same_term} and kind in ("condition", "symptom"):
            reasons.append("NEGATION")

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return None, reasons, len(nums)
    flags = []
    odd = sorted({w for w in _words(text) if w not in _CONNECTORS and w not in vocab and w not in src_words and _stem(w) not in src_words})
    if odd:
        flags.append({"code": "unverified_words", "words": odd[:6], "detail": "Words that are not in the note or the facts. Check them."})
    return Sentence(text, list(dict.fromkeys(ids)), flags), [], len(nums)


def validate_output(content: str, facts: list[dict], source_text: str) -> Validation:
    items = parse_output(content)
    by_id = {f["id"]: f for f in facts}
    src_numbers, src_words = llm_extract.numbers(source_text), _words(source_text)
    src_words |= {_stem(w) for w in src_words}
    vocab = set()
    for f in facts:
        vocab |= _words(" ".join(str(f.get(k) or "") for k in ("term", "mtype", "unit", "food", "kind", "status")))
    vocab |= {_stem(w) for w in vocab}
    accepted, reasons, checked, rejected = [], Counter(), 0, 0
    for it in items:
        s, why, n = validate_sentence(it, by_id, src_numbers, src_words, vocab)
        checked += n
        if s:
            accepted.append(s)
        else:
            rejected += 1
            reasons.update(why)
    return Validation(accepted, reasons, checked, rejected)


def coverage(facts: list[dict], accepted: list[Sentence]) -> dict:
    used = {i for s in accepted for i in s.facts}
    left = [f["id"] for f in facts if f["id"] not in used]
    return {"facts": len(facts), "used": len(used), "not_mentioned": left}


# ---------------------------------------------------------------- pending clinician review (in memory; ids and counts only)
class PendingReviews:
    """Remembers that a summary was produced and is waiting for review: request id, the key that made it, a timestamp and counts. Never text.
    In memory and per process, like the rate limiters: a restart or an expiry means the summary must be generated again."""

    def __init__(self, ttl_s: int, max_items: int = 1000):
        self.ttl, self.max, self._d, self._lock = ttl_s, max_items, {}, threading.Lock()

    def add(self, request_id: str, key_id: int, **counts) -> None:
        now = time.time()
        with self._lock:
            self._d = {k: v for k, v in self._d.items() if now - v["at"] < self.ttl}
            while len(self._d) >= self.max:
                self._d.pop(next(iter(self._d)))
            self._d[request_id] = {"key_id": key_id, "at": now, **counts}

    def take(self, request_id: str, key_id: int):
        """The pending entry (removed: a summary is reviewed once), or None when unknown, expired, already reviewed or made by another key."""
        now = time.time()
        with self._lock:
            e = self._d.get(request_id)
            if not e or now - e["at"] >= self.ttl or e["key_id"] != key_id:
                return None
            del self._d[request_id]
            return {**e, "age_s": int(now - e["at"])}


pending = PendingReviews(3600)
