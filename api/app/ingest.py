"""Dataset ingestion: safe parsing, cleaning, profiling, de-identification.

Design rules
- Nothing is imputed or encoded here. Imputation/encoding belong inside the
  training pipeline so they are fitted on training folds only.
- The raw file is never stored. Flagged columns are never profiled or echoed.
- Direct identifiers cannot be kept without an explicit, audited reason.
"""
import hashlib
import hmac
import io
import re
import zipfile
from dataclasses import dataclass, field

import pandas as pd

from .config import get_settings

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "unknown", "#n/a"}
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
ANY_DATE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?265[\s-]?|0)[1-9]\d{1,2}[\s-]?\d{3}[\s-]?\d{3,4}(?!\d)")
LONG_DIGITS = re.compile(r"^\d{9,13}$")


class IngestError(ValueError):
    """Raised with a message that is safe to show to the user."""


# ---------- parsing ----------
def parse_upload(raw: bytes, filename: str) -> tuple[pd.DataFrame, str, list[str]]:
    s = get_settings()
    warnings: list[str] = []
    ext = (filename or "").rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if ext not in ("csv", "xlsx"):
        raise IngestError("Only .csv and .xlsx files are accepted (.xls and macro-enabled files are not).")
    is_zip = raw[:2] == b"PK"
    if ext == "xlsx" and not is_zip:
        raise IngestError("The file is not a valid .xlsx workbook.")
    if ext == "csv" and is_zip:
        raise IngestError("The file has a .csv name but is an Excel/zip file. Re-save it as CSV.")

    if ext == "xlsx":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                if sum(i.file_size for i in z.infolist()) > 25 * s.max_upload_mb * 1024 * 1024:
                    raise IngestError("The workbook expands to an unsafe size.")
                if any(i.filename.lower().endswith("vbaproject.bin") for i in z.infolist()):
                    raise IngestError("Workbooks containing macros are not accepted.")
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, dtype=str, keep_default_na=False,
                                   engine="openpyxl", nrows=s.max_rows + 1)
        except IngestError:
            raise
        except Exception:
            raise IngestError("The workbook could not be read. Check that it is a valid, unprotected .xlsx file.")
        if not sheets:
            raise IngestError("The workbook has no sheets.")
        names = list(sheets)
        df = sheets[names[0]]
        if len(names) > 1:
            warnings.append(f"Workbook has {len(names)} sheets; only the first was used.")
    else:
        text = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None or not text.strip():
            raise IngestError("The CSV file is empty or unreadable.")
        sep = ","
        head = text[:8192]
        counts = {c: head.count(c) for c in (",", ";", "\t", "|")}
        if max(counts.values()) > 0:
            sep = max(counts, key=counts.get)
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False,
                             nrows=s.max_rows + 1, on_bad_lines="error")
        except Exception:
            raise IngestError("The CSV could not be parsed. Check for inconsistent column counts or stray quotes.")

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise IngestError("The file has no data rows.")
    if df.shape[0] > s.max_rows:
        raise IngestError(f"Too many rows (limit {s.max_rows:,}).")
    if df.shape[1] > s.max_columns:
        raise IngestError(f"Too many columns (limit {s.max_columns}).")
    df.columns = _normalise_names(list(df.columns))
    return df, ext, warnings


def _normalise_names(cols: list) -> list[str]:
    out, seen = [], {}
    for i, c in enumerate(cols):
        n = re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_") or f"column_{i + 1}"
        n = n[:80]
        if n[0].isdigit():
            n = "c_" + n
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


# ---------- cleaning / typing ----------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trim, standardise missing tokens, infer types. Returns (df, notes)."""
    notes = {"coerced_to_missing": {}, "date_assumptions": {}}
    df = df.copy()
    for c in df.columns:
        col = df[c].astype("string").str.strip()
        col = col.mask(col.str.lower().isin(MISSING_TOKENS))
        df[c] = col
    for c in df.columns:
        col, non_null = df[c], df[c].dropna()
        if non_null.empty:
            continue
        num = pd.to_numeric(non_null.str.replace(",", "", regex=False), errors="coerce")
        if num.notna().mean() >= 0.95 and not _looks_like_code(non_null):
            full = pd.to_numeric(col.str.replace(",", "", regex=False), errors="coerce")
            bad = int(full.isna().sum() - col.isna().sum())
            if bad:
                notes["coerced_to_missing"][c] = bad
            fl = full.astype("float64")
            df[c] = fl.astype("Int64") if (fl.dropna() % 1 == 0).all() else fl
            continue
        sample = non_null.head(500)
        if sample.str.match(ANY_DATE).mean() >= 0.95:
            iso = non_null.str.match(ISO_DATE).all()
            parsed = pd.to_datetime(col, errors="coerce", dayfirst=not iso, format="%Y-%m-%d" if iso and non_null.str.len().max() == 10 else "mixed")
            if parsed.notna().sum() >= 0.95 * len(non_null):
                bad = int(parsed.isna().sum() - col.isna().sum())
                if bad:
                    notes["coerced_to_missing"][c] = bad
                df[c] = parsed
                notes["date_assumptions"][c] = "ISO year-month-day" if iso else "day first (dd/mm/yyyy)"
    return df, notes


def _looks_like_code(s: pd.Series) -> bool:
    # Leading zeros ("007") or very long digit strings are identifiers/codes, not quantities.
    sample = s.head(500)
    return bool(sample.str.match(r"^0\d+$").mean() > 0.2 or sample.str.match(LONG_DIGITS).mean() > 0.5)


# ---------- identifier detection ----------
NAME_TOKENS = {"name", "firstname", "lastname", "surname", "fullname", "forename", "initials"}
NAME_SUBSTR = ("mother", "father", "guardian", "caregiver", "next_of_kin", "nok", "kin_")
CONTACT_TOKENS = {"phone", "tel", "telephone", "mobile", "cell", "whatsapp", "email", "mail", "contact"}
ADDRESS_TOKENS = {"address", "village", "street", "gps", "latitude", "longitude", "lat", "lon", "lng", "coordinates", "postcode", "zip", "compound"}
ID_TOKENS = {"nid", "passport", "mrn", "ssn"}
ID_PHRASES = ("national_id", "patient_id", "patient_no", "patient_number", "hospital_no", "hospital_number",
              "record_no", "record_number", "registration_no", "reg_no", "file_no", "folder_no", "id_number", "id_no")
DOB_PHRASES = ("dob", "date_of_birth", "birth_date", "birthdate", "birthday")
QUASI_TOKENS = {"sex", "gender", "district", "region", "facility", "ta", "age", "year", "month"}


@dataclass
class ColumnReport:
    name: str
    kind: str                       # numeric | date | categorical | text
    category: str = "none"          # none | person_name | contact | address | identifier | date_of_birth | free_text | value_pii
    reason: str = ""
    suggested: str = "keep"
    quasi: bool = False
    can_keep: bool = True           # False => never keepable (PII found in the values)
    profile: dict = field(default_factory=dict)


def _tokens(name: str) -> set[str]:
    return set(name.split("_"))


def _classify_name(name: str) -> tuple[str, str] | None:
    t = _tokens(name)
    if any(p in name for p in DOB_PHRASES):
        return "date_of_birth", "Looks like a date of birth."
    if t & NAME_TOKENS or any(p in name for p in NAME_SUBSTR):
        return "person_name", "Column name suggests a person's name."
    if t & CONTACT_TOKENS:
        return "contact", "Column name suggests contact details."
    if t & ADDRESS_TOKENS:
        return "address", "Column name suggests a precise location."
    if t & ID_TOKENS or any(p in name for p in ID_PHRASES):
        return "identifier", "Column name suggests a personal or record identifier."
    if name == "id" or name.endswith("_id"):
        return "identifier", "Looks like a record identifier."
    return None


def _value_pii(col: pd.Series) -> str | None:
    vals = col.dropna()
    if vals.empty:
        return None
    sample = vals.astype(str).head(5000)
    n = len(sample)
    if sample.str.contains(EMAIL).sum() / n >= 0.005:
        return "Values contain email addresses."
    if sample.str.contains(PHONE).sum() / n >= 0.02:
        return "Values contain phone numbers."
    if pd.api.types.is_integer_dtype(col) and sample.str.fullmatch(LONG_DIGITS).mean() > 0.5:
        return "Values look like phone or ID numbers."
    return None


def analyse(df: pd.DataFrame) -> list[ColumnReport]:
    out: list[ColumnReport] = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            kind = "date"
        elif pd.api.types.is_numeric_dtype(col):
            kind = "numeric"
        else:
            kind = "categorical"
            nn = col.dropna().astype(str)
            if not nn.empty and (nn.str.len().mean() > 60 or nn.str.split().str.len().mean() > 8):
                kind = "text"
        r = ColumnReport(name=c, kind=kind)
        hit = _classify_name(c)
        if hit:
            r.category, r.reason = hit
        value_hit = _value_pii(col) if kind in ("categorical", "text", "numeric") else None
        if value_hit:
            r.category, r.reason, r.can_keep = "value_pii", value_hit, False
        elif r.category == "none" and kind == "text":
            r.category, r.reason = "free_text", "Free text can contain names or other identifiers."
        # defaults
        if r.category == "date_of_birth":
            r.suggested = "year_only" if kind == "date" else "drop"
        elif r.category == "identifier":
            r.suggested = "drop"
        elif r.category != "none":
            r.suggested = "drop"
        # quasi-identifiers (used for the k-anonymity check)
        r.quasi = bool(_tokens(c) & QUASI_TOKENS) or kind == "date" or r.category == "date_of_birth"
        if r.category == "none" and kind == "date":
            r.reason = "Full dates are quasi-identifiers; consider keeping year only."
        # Profile only columns that are safe to show
        if r.category == "none" or r.category == "date_of_birth":
            r.profile = profile_column(col, kind)
        else:
            r.profile = {"missing": int(col.isna().sum()), "hidden": True}
        out.append(r)
    return out


def profile_column(col: pd.Series, kind: str) -> dict:
    n = len(col)
    miss = int(col.isna().sum())
    p: dict = {"missing": miss, "missing_pct": round(100 * miss / n, 1) if n else 0.0, "unique": int(col.nunique(dropna=True))}
    nn = col.dropna()
    if nn.empty:
        return p
    if kind == "numeric":
        p.update(min=_num(nn.min()), max=_num(nn.max()), mean=_num(nn.mean()), median=_num(nn.median()),
                 std=_num(nn.std()) if len(nn) > 1 else None)
    elif kind == "date":
        p.update(min=str(nn.min().date()), max=str(nn.max().date()))
    else:
        vc = nn.astype(str).value_counts()
        shown = vc[vc >= 5].head(10)
        p["top"] = [{"value": str(k), "count": int(v)} for k, v in shown.items()]
        suppressed = int(vc[vc < 5].sum())
        if suppressed:
            p["suppressed_small_counts"] = suppressed  # cells < 5 are not shown
    return p


def _num(x) -> float | None:
    try:
        return None if pd.isna(x) else round(float(x), 4)
    except (TypeError, ValueError):
        return None


def default_plan(reports: list[ColumnReport]) -> dict[str, str]:
    return {r.name: r.suggested for r in reports}


# ---------- applying a plan ----------
def pseudonymise(series: pd.Series, column: str) -> pd.Series:
    key = get_settings().pseudonym_key
    if len(key) < 32:
        raise IngestError("Pseudonymisation is not enabled on this server (PSEUDONYM_KEY not set).")
    kb = key.encode()
    return series.map(lambda v: None if pd.isna(v) else
                      hmac.new(kb, f"{column}:{v}".encode(), hashlib.sha256).hexdigest()[:16])


def apply_plan(df: pd.DataFrame, reports: list[ColumnReport], actions: dict[str, tuple[str, str | None]],
               drop_duplicates: bool, notes: dict) -> tuple[pd.DataFrame, dict, dict]:
    """actions: column -> (action, override_reason). Returns (df, report, schema)."""
    by = {r.name: r for r in reports}
    rep: dict = {"rows_in": int(len(df)), "columns_in": int(df.shape[1]), "dropped": [], "pseudonymised": [],
                 "generalised": [], "overrides": [], "warnings": [],
                 "coerced_to_missing": notes.get("coerced_to_missing", {}),
                 "date_assumptions": notes.get("date_assumptions", {})}
    out = df.copy()
    for name, (action, reason) in actions.items():
        r = by[name]
        if action == "keep":
            if not r.can_keep:
                raise IngestError(f"Column '{name}' contains personal data and cannot be kept. Drop it or pseudonymise it.")
            if r.category != "none":
                if not reason or len(reason.strip()) < 10:
                    raise IngestError(f"Column '{name}' was flagged ({r.reason.rstrip('.')}). Keeping it needs a written reason (10+ characters).")
                rep["overrides"].append({"column": name, "category": r.category, "reason": reason.strip()})
        elif action == "pseudonymise":
            if r.kind in ("date", "text") or r.category == "free_text":
                raise IngestError(f"Column '{name}' cannot be pseudonymised (dates and free text must be dropped or generalised).")
            out[name] = pseudonymise(out[name].astype("string"), name)
            rep["pseudonymised"].append(name)
        elif action == "year_only":
            if r.kind != "date":
                raise IngestError(f"'year_only' applies to date columns; '{name}' is not one.")
            out[name] = out[name].dt.year.astype("Int64")
            rep["generalised"].append({"column": name, "to": "year"})
        elif action == "drop":
            out = out.drop(columns=[name])
            rep["dropped"].append({"column": name, "reason": r.reason or "Dropped by the data scientist."})
    empty = int(out.isna().all(axis=1).sum())
    if empty:
        out = out.dropna(how="all")
    rep["empty_rows_removed"] = empty
    dup = 0
    if drop_duplicates and out.shape[1]:
        before = len(out)
        out = out.drop_duplicates()
        dup = before - len(out)
    rep["duplicates_removed"] = int(dup)
    if out.shape[1] == 0 or len(out) == 0:
        raise IngestError("Nothing is left after applying the plan.")
    rep["rows_out"], rep["columns_out"] = int(len(out)), int(out.shape[1])
    rep["k_anonymity"] = k_anonymity(out, [r.name for r in reports if r.quasi and r.name in out.columns])
    if rep["k_anonymity"].get("k_min") is not None and rep["k_anonymity"]["k_min"] < 5:
        rep["warnings"].append("Some combinations of quasi-identifiers occur fewer than 5 times; individuals may be re-identifiable. Generalise or drop columns before wider use.")
    if rep["overrides"]:
        rep["warnings"].append("Columns that look like identifiers were kept with a recorded reason.")
    kinds = {r.name: r.kind for r in reports}
    schema = {"columns": [{"name": c, "kind": "numeric" if pd.api.types.is_numeric_dtype(out[c]) else
                           "date" if pd.api.types.is_datetime64_any_dtype(out[c]) else kinds.get(c, "categorical"),
                           **profile_column(out[c], "numeric" if pd.api.types.is_numeric_dtype(out[c]) else
                                            "date" if pd.api.types.is_datetime64_any_dtype(out[c]) else "categorical")}
                          for c in out.columns]}
    return out, rep, schema


def k_anonymity(df: pd.DataFrame, quasi: list[str]) -> dict:
    quasi = [c for c in quasi if c in df.columns]
    if not quasi:
        return {"columns": [], "k_min": None, "rows_in_small_groups": 0}
    g = df.groupby(quasi, dropna=False, observed=True).size()
    return {"columns": quasi, "k_min": int(g.min()), "rows_in_small_groups": int(g[g < 5].sum())}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
