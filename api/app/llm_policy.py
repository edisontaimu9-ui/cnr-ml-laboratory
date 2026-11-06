"""LLM intelligence, phase 1: the safety rules every later LLM feature must pass through.

Four rules, all enforced in code rather than left to a prompt:

1. DECISION SUPPORT ONLY. The laboratory never diagnoses. A question that asks it to diagnose, or to prescribe for a named kind of
   patient, is refused (`screen_request`); an answer that states a diagnosis or a prescription about a person is blocked before it
   is returned (`screen_output`). Explaining criteria, extracting what a document says and retrieving guideline sources stay allowed.
2. NOTHING IDENTIFIABLE LEAVES THE LABORATORY. Text goes to an external service only after the NLP identifier detector has run on it
   (`prepare_for_external`): identifiable text is refused (default) or masked. Detection is conservative but incomplete (untitled
   personal names, addresses and dates of birth are not found), so callers must not treat "no identifiers found" as "safe".
3. NO CLINICAL TEXT IN THE AUDIT LOG. `llm_audit.record` accepts counts, booleans and short fixed codes only.
4. LIMITS PER API KEY (see llm_limits.py).

These checks are rule-based and transparent. They are a first and a last line of defence around a model, not a guarantee about what a
model says; clinician review (phase 4) remains the control for anything used on a patient. Only English patterns are covered by the
request and output screens so far; no Chichewa, Tumbuka or Yao patterns have been written or clinically reviewed.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from . import nlp_lab
from . import nlp_lexicon as lx

POLICY_VERSION = "cnr-llm-policy/1"
POLICY_CODE = "cnr-llm-policy-1"       # the same, in the shape the audit allow-list accepts
CONTRACT = "cnr-llm-contract/1"

NOTICE = ("Decision support only. This service does not diagnose, prescribe or replace clinical judgement; a qualified clinician must "
          "review anything used for a patient. Text is not stored, and identifiable text is refused or masked before external processing.")

# What a later phase must put in front of any model it calls. The tests assert that these lines stay.
SYSTEM_RULES = (
    "You are a decision-support assistant for nutrition and dietetics in Malawi.\n"
    "Rules you must follow without exception:\n"
    "1. Never diagnose. Never say that a person has, or probably has, a disease or nutritional condition. Describe criteria, "
    "documented findings and guideline content, and say that a clinician makes the diagnosis.\n"
    "2. Never prescribe. Never give a drug or supplement dose for a particular person. Quote guideline doses only as guideline text, "
    "with the source.\n"
    "3. Use only the sources and facts you are given. If they do not answer the question, say so. Never invent a number, a source or a "
    "guideline.\n"
    "4. Never ask for, repeat or guess personal identifiers (names, phone numbers, record numbers, addresses).\n"
    "5. If someone may be in danger, tell them to seek urgent clinical care now.\n")

RULES = [
    "Decision support only: the service never diagnoses and never prescribes for a person.",
    "Questions that ask for a diagnosis or a patient-specific prescription are refused; criteria, guidelines and documented facts may be explained.",
    "Answers that state a diagnosis or prescription about a person are blocked before they are returned.",
    "Text leaves the laboratory only after the identifier detector has run; identifiable text is refused (default) or masked.",
    "The audit log holds counts and fixed codes only, never clinical text, questions, answers or identifiers.",
    "The Chakudya API address and consumer key are held on the server only and are never sent to a browser or returned by an endpoint.",
    "Every request is limited per API key.",
]

LIMITATIONS = [
    "Identifier detection finds e-mail addresses, phone numbers, record and ID numbers and titled names. It does not find untitled "
    "personal names, addresses or dates of birth.",
    "The diagnosis and prescription screens are English-only pattern rules. They reduce, and do not remove, the chance of a boundary breach.",
    "Rate limits are kept in memory, per process.",
]

CATEGORIES = {
    "diagnosis_request": "The question asks the service to diagnose a person. It can explain diagnostic criteria and what a document states, "
                         "but the diagnosis is for a clinician.",
    "prescribing_request": "The question asks the service to prescribe or dose for a particular patient. It can quote guideline text with its "
                           "source, but prescribing is for a clinician.",
    "identifiers_found": "The text contains personal identifiers. Remove them at the source and send the text again, or ask for masking.",
    "empty": "The text is empty.",
    "too_long": "The text is longer than the limit.",
    "output_diagnosis": "The answer stated a diagnosis about a person, which this service does not do.",
    "output_prescription": "The answer gave a prescription or dose for a person, which this service does not do.",
}


class PolicyRefusal(Exception):
    def __init__(self, category: str, message: str | None = None, **details):
        self.category, self.message, self.details = category, message or CATEGORIES.get(category, category), details
        super().__init__(self.message)


# ---------------------------------------------------------------- 1. decision-support boundary (requests)
_PERSON = (r"(?:he|she|they|i|me|my\s+\w+|the\s+(?:patient|child|baby|infant|toddler|mother|client|woman|man|boy|girl|person)|"
           r"this\s+(?:patient|child|baby|infant|toddler|person|client|woman|man|boy|girl)|(?:patient|client))")
_COND = "|".join(re.escape(t) for t in sorted({e["term"] for e in lx.entries() if e["label"] == "CONDITION" and e["lang"] in ("en", "both")},
                                                key=len, reverse=True))
_HAS = r"(?:have|has|got|suffer(?:s|ing)?\s+from|developed?)"
_RX = lambda p: re.compile(p, re.I)  # noqa: E731

_DIAGNOSIS = [
    _RX(r"\bdiagnos(?:e|ing)\s+(?:this|that|my|the|him|her|them|me|a|an|our)\b"),
    _RX(r"\b(?:can|could|will|would)\s+you\s+diagnos(?:e|is)\b"),
    _RX(r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?diagnosis\b.{0,40}\b(?:for|of)\s+" + _PERSON + r"\b"),
    _RX(r"\bwhat(?:'s|\s+is)\s+(?:wrong|the\s+matter)\s+with\s+" + _PERSON + r"\b"),
    _RX(r"\b(?:what|which)\s+(?:disease|illness|condition|disorder|deficiency|malnutrition)\s+(?:does|do|did|is|has|have)\b.{0,30}\b" + _PERSON + r"\b"),
    _RX(r"\b(?:does|do|did|is|are|am|could|might|may|can)\s+" + _PERSON + r"\s+" + _HAS + r"\b(?:\s+\w+){0,4}?\s+(?:" + _COND + r")\b"),
    _RX(r"\b(?:is|are|am)\s+" + _PERSON + r"\s+(?:malnourished|anaemic|anemic|diabetic|hypertensive|stunted|wasted|underweight|obese|severely\s+malnourished)\b"),
    _RX(r"\b(?:confirm|tell\s+me|determine|decide|say|check)\s+(?:if|whether|that)\s+" + _PERSON + r"\s+(?:has|have|is|are)\b"),
]
_PRESCRIBING = [
    _RX(r"\bprescribe\b"),
    _RX(r"\bwhat\s+(?:dose|dosage)\b.{0,30}\b(?:for|should)\b.{0,30}\b" + _PERSON + r"\b"),
    _RX(r"\b(?:dose|dosage)\s+(?:for|of)\b.{0,40}\b(?:my|the|this)\s+(?:patient|child|baby|infant|toddler|client|mother)\b"),
    _RX(r"\bhow\s+(?:much|many)\s+(?:mg|ml|tablets?|capsules?|drops|units?|sachets?)\b.{0,40}\b(?:should|do)\s+(?:i|we)\s+(?:give|administer|prescribe|start)\b"),
    _RX(r"\bwhat\s+(?:drug|medicine|medication|antibiotic|supplement)s?\s+should\s+(?:i|we)\s+(?:give|start|prescribe|use)\b"),
]


def screen_request(text: str) -> str | None:
    """None when the question is allowed, otherwise the refusal category."""
    t = " ".join(sanitize(text).split())
    if any(rx.search(t) for rx in _DIAGNOSIS):
        return "diagnosis_request"
    if any(rx.search(t) for rx in _PRESCRIBING):
        return "prescribing_request"
    return None


# ---------------------------------------------------------------- 1b. decision-support boundary (answers)
_SUBJECT = r"(?:the\s+(?:patient|child|baby|infant|toddler|mother|client|woman|man|boy|girl|person)|this\s+(?:patient|child|baby|infant|toddler|person|client|woman|man|boy|girl)|your\s+(?:child|baby|patient|condition)|you|he|she)"
_OUT_DIAGNOSIS = [
    _RX(r"\b" + _SUBJECT + r"\s+(?:has|have|had|is\s+suffering\s+from|are\s+suffering\s+from|suffers?\s+from|is\s+diagnosed\s+with|was\s+diagnosed\s+with)\b(?:\s+(?:a|an|the|severe|moderate|mild|acute|chronic|clear|definite|confirmed))*\s+(?:" + _COND + r")\b"),
    _RX(r"\b" + _SUBJECT + r"\s+(?:is|are)\s+(?:definitely\s+|clearly\s+|certainly\s+)?(?:malnourished|severely\s+malnourished|anaemic|anemic|diabetic|hypertensive|stunted|wasted|obese|dehydrated)\b"),
    _RX(r"\b(?:the\s+diagnosis\s+is|my\s+diagnosis\s+is|i\s+(?:can\s+)?(?:diagnose|confirm)|diagnosis:\s)"),
    _RX(r"\b(?:definitely|certainly|clearly)\s+(?:has|have|is)\s+(?:" + _COND + r")\b"),
]
_OUT_PRESCRIPTION = [
    _RX(r"\b(?:i\s+(?:would\s+)?(?:prescribe|recommend\s+(?:giving|administering))|you\s+should\s+(?:take|give|administer)|give\s+(?:the\s+)?(?:patient|child|baby|infant|him|her)|administer)\s+(?:\w+\s+){0,3}?\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|g|iu|units?|tablets?|capsules?|drops|sachets?)\b"),
    _RX(r"\b(?:i\s+prescribe|prescribed\s+for\s+(?:you|the\s+patient|this\s+child))\b"),
]
REFUSED_OUTPUT = ("This answer was withheld because it stated a diagnosis or a prescription about a person, which this service does not "
                  "provide. A clinician makes diagnoses and prescribes. Ask about the criteria, the guideline or what a document says instead.")


@dataclass
class OutputVerdict:
    allowed: bool
    categories: list[str] = field(default_factory=list)


def screen_output(text: str) -> OutputVerdict:
    t = " ".join(sanitize(text).split())
    cats = []
    if any(rx.search(t) for rx in _OUT_DIAGNOSIS):
        cats.append("output_diagnosis")
    if any(rx.search(t) for rx in _OUT_PRESCRIPTION):
        cats.append("output_prescription")
    return OutputVerdict(allowed=not cats, categories=cats)


def enforce_output(text: str) -> tuple[str, OutputVerdict]:
    """The text to return (the fixed refusal text when blocked) and the verdict."""
    v = screen_output(text)
    return (text if v.allowed else REFUSED_OUTPUT), v


# ---------------------------------------------------------------- 2. nothing identifiable leaves the laboratory
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad\u202a-\u202e\u2066-\u2069]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize(text: str) -> str:
    """Compatibility-fold (full-width digits become ASCII) and drop zero-width and control characters, so identifiers cannot slip past
    the detector by being written with look-alike or invisible characters. The sanitized text is what is detected on, masked and sent."""
    return _CONTROL.sub("", _INVISIBLE.sub("", unicodedata.normalize("NFKC", text)))


@dataclass
class Prepared:
    text: str                      # the only text that may be sent to an external service
    masked: bool
    identifiers_found: int
    identifier_types: dict[str, int]
    chars_in: int
    warnings: list[str] = field(default_factory=list)


def prepare_for_external(text: str, mode: str, *, max_chars: int) -> Prepared:
    """Run the NLP identifier detector. Raises PolicyRefusal for empty, over-long or (mode "reject") identifiable text. With mode "mask",
    identifiers are replaced and the masked text is checked again before it is returned."""
    if mode not in ("mask", "reject"):
        raise ValueError("mode must be 'mask' or 'reject'")
    if not isinstance(text, str) or not text.strip():
        raise PolicyRefusal("empty")
    if len(text) > max_chars:
        raise PolicyRefusal("too_long", f"The text is longer than {max_chars} characters.", limit=max_chars)
    clean = sanitize(text)
    phi = nlp_lab.screen_identifiers(clean)
    types: dict[str, int] = {}
    for e in phi:
        types[e["norm"]] = types.get(e["norm"], 0) + 1
    warnings = ["Untitled personal names, addresses and dates of birth are not detected; do not rely on this check alone."]
    if not phi:
        return Prepared(clean, False, 0, {}, len(text), warnings)
    if mode == "reject":
        raise PolicyRefusal("identifiers_found", found=len(phi), types=types)
    out = nlp_lab.mask_identifiers(clean, phi)
    if nlp_lab.screen_identifiers(out):    # defence in depth: masking must leave nothing the detector still finds
        raise PolicyRefusal("identifiers_found", "Identifiers remain after masking; remove them at the source.", found=len(phi), types=types)
    warnings.insert(0, "Identifiers were masked before processing.")
    return Prepared(out, True, len(phi), types, len(text), warnings)


# ---------------------------------------------------------------- response envelope
def envelope(**fields) -> dict:
    """Common fields of every LLM response: policy version and the decision-support statement. Callers add their own fields."""
    return {"contract": CONTRACT, "policy": POLICY_VERSION, "decision_support_only": True, "stored": False, "notice": NOTICE, **fields}
