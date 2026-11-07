"""Serving for deployed NLP models (entity model, language identification, document-type classifier).

Only a version whose status is `deployed` is ever loaded, one per model name, and its stored file is checked against the hash recorded
at registration before it is loaded. A model is used only for the languages a reviewer approved: for other languages the entity and
classifier endpoints fall back to the rule engine and say so, and the language identifier flags a prediction it was not reviewed for.
Text is screened for identifiers first and refused, never echoed. Inputs and outputs are not stored; the audit log keeps counts only.
Loaded models are cached per (version id, file hash) and dropped when anything is deployed, rolled back or retired."""
import hashlib
import threading

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import nlp_lab, nlp_textmodels, nlp_training, storage
from .models import NlpModelVersion

_lock = threading.Lock()
_cache: dict[str, tuple[int, str, object]] = {}
NOTICE = ("Decision support only. This model was approved by a reviewer for the languages listed and for its intended use; it is not "
          "clinically validated. Check its output before relying on it.")


class NlpServingError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def kind_of(v: NlpModelVersion) -> str:
    m = v.metrics_json or {}
    return m.get("task") or ("ner" if str(m.get("engine", "")).startswith("spacy") else "ner")


def deployed(db: Session, name: str) -> NlpModelVersion:
    v = db.scalar(select(NlpModelVersion).where(NlpModelVersion.name == name, NlpModelVersion.status == "deployed"))
    if v is None:
        raise NlpServingError(404, "No model with that name is deployed.")
    return v


def _load(v: NlpModelVersion):
    with _lock:
        hit = _cache.get(v.name)
        if hit and hit[0] == v.id and hit[1] == v.artifact_sha256:
            return hit[2]
    try:
        data = storage.load_bytes(v.artifact_uri) if v.artifact_uri and storage.exists(v.artifact_uri) else None
    except Exception:
        data = None
    if data is None or hashlib.sha256(data).hexdigest() != v.artifact_sha256:
        raise NlpServingError(409, "The stored model file is missing or does not match its recorded hash, so it will not be loaded. Investigate before continuing.")
    try:
        obj = nlp_training.load_model(data) if kind_of(v) == "ner" else nlp_textmodels.load_model(data)
    except Exception:
        raise NlpServingError(409, "The model file could not be loaded.")
    with _lock:
        _cache[v.name] = (v.id, v.artifact_sha256, obj)
    return obj


def block(v: NlpModelVersion) -> dict:
    m = v.metrics_json or {}
    return {"name": v.name, "version": v.version, "version_id": v.id, "kind": kind_of(v), "engine": m.get("engine"),
            "approved_languages": v.approved_languages_json or [], "approved_by": v.approved_by, "deployed_at": v.deployed_at.isoformat() + "Z" if v.deployed_at else None,
            "intended_use": v.intended_use, "clinical_validation_ref": v.clinical_validation_ref,
            "evaluation": {"overall_f1": ((m.get("model") or {}).get("overall") or {}).get("f1"), "macro_f1": (m.get("model") or {}).get("macro_f1"),
                           "rules_macro_f1": (m.get("rules") or {}).get("macro_f1"), "test_documents": (m.get("split") or {}).get("test_documents")}}


def predict(v: NlpModelVersion, text: str, lang: str | None) -> dict:
    """Raises NlpServingError for identifiers. Returns the result part of the response."""
    if nlp_lab.screen_identifiers(text):
        raise NlpServingError(422, "Identifiers were found in the text. Remove them and send it again.")
    kind, approved = kind_of(v), set(v.approved_languages_json or [])
    a = nlp_lab.analyze(text)
    if kind == "langid":
        model, meta = _load(v)
        out = nlp_textmodels.predict(model, meta, text)
        out.update(kind=kind, model_used=True, reviewed=out["label"] in approved)
        if not out["reviewed"]:
            out["reason"] = f"The model was not approved for {out['label']}; treat this answer as unreviewed."
        return out
    guessed = lang or a["language"]["overall"]
    source = "given" if lang else "rules"
    base = {"kind": kind, "language": {"code": guessed, "source": source}}
    if guessed not in approved:
        reason = (f"The model is not approved for {guessed}" if guessed != "unknown" else "The language could not be determined; pass lang") + ", so the rule engine answered instead."
        fallback = ({"entities": [{"start": e["start"], "end": e["end"], "label": e["label"], "text": e["text"]} for e in a["entities"] if e["label"] != "PHI"]}
                    if kind == "ner" else {"label": a["classification"]["document_type"]["label"], "probability": None, "ranked": []})
        return {**base, **fallback, "model_used": False, "reason": reason, "answered_by": nlp_lab.ENGINE_VERSION}
    model = _load(v)
    if kind == "ner":
        doc = model(text)
        return {**base, "entities": [{"start": e.start_char, "end": e.end_char, "label": e.label_, "text": e.text} for e in doc.ents],
                "model_used": True, "answered_by": f"{v.name} v{v.version}"}
    pipe, meta = model
    return {**base, **nlp_textmodels.predict(pipe, meta, text), "model_used": True, "answered_by": f"{v.name} v{v.version}"}
