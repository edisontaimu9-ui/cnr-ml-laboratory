"""Food-name matching: evaluation of a learned matcher from free-text names (English, Chichewa, local or
colloquial) to registry food codes.

This is an evaluation tool, not a served model. Like explorations it has no model definition, writes no model
file and imports nothing from the registry or serving code, so nothing here can be registered, deployed or
sent to CNR. Only written results come back, and they hold counts and rates, never the query text.

Method (kept deliberately simple and inspectable):
  * every registry food is represented by its name, its known aliases and, for the learned matcher, the labelled
    queries seen in training (a labelled query becomes one more way to write that food);
  * a query is compared with all of them by cosine similarity of character 2-4-grams (TF-IDF), which absorbs
    spelling variation, and a food scores the best of its entries;
  * the score is a similarity, not a probability. Below a threshold chosen on training data only, the matcher
    abstains and the query goes to a person.

Honest evaluation: queries are split by normalised text, so the same string can never sit in both train and
test; a names-only matcher and an exact-match rule are scored on the same test rows; the learned matcher is
credited only if the paired bootstrap says it beats names-only; queries that are already a registry name are
reported apart so they cannot flatter the score.
"""
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

MIN_PAIRS = 50
MIN_FOODS = 20
MIN_TEST = 20
MAX_QUERY_LEN = 120
MIN_GROUP_TEST = 10
CHUNK = 512
BOOTSTRAP = 1000
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.30, 0.96, 0.05)]
MIN_ANSWERED_FOR_THRESHOLD = 20
ALIAS_SPLIT = re.compile(r"[;|]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class MatchingError(Exception):
    pass


@dataclass
class Config:
    query_col: str
    food_col: str
    registry_id_col: str
    registry_name_col: str
    registry_alias_col: str | None = None
    group_col: str | None = None
    test_size: float = 0.25
    seed: int = 42
    top_k: int = 5
    target_accuracy: float = 0.95


def normalise(s) -> str:
    """Lower case, accents removed, anything that is not a letter or digit becomes one space."""
    t = unicodedata.normalize("NFKD", str(s))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return _NON_ALNUM.sub(" ", t.lower()).strip()


def _r(x, n: int = 4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


class Index:
    """Char-n-gram TF-IDF index over registry names and aliases (`base`) plus optional learned queries."""

    def __init__(self, ids: list[str], base: list[list[str]], learned: dict[int, list[str]] | None = None):
        self.ids = ids
        texts, owner, exact = [], [], {}
        for i, names in enumerate(base):
            seen: set[str] = set()
            for t in names:
                n = normalise(t)
                if n:
                    exact.setdefault(n, set()).add(i)
                    if n not in seen:
                        seen.add(n)
                        texts.append(n)
                        owner.append(i)
            for t in (learned or {}).get(i, []):
                n = normalise(t)
                if n and n not in seen:
                    seen.add(n)
                    texts.append(n)
                    owner.append(i)
        order = np.argsort(np.asarray(owner), kind="stable")
        texts, self.owner = [texts[j] for j in order], np.asarray(owner)[order]
        if len(set(self.owner.tolist())) != len(ids):
            raise MatchingError("Every registry food needs a name that contains letters or digits.")
        self.starts = np.flatnonzero(np.r_[True, self.owner[1:] != self.owner[:-1]])
        self.exact = exact
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        self.E = self.vec.fit_transform(texts)

    def top_k(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.ids))
        idx = np.zeros((len(queries), k), dtype=int)
        sc = np.zeros((len(queries), k), dtype=float)
        for s in range(0, len(queries), CHUNK):
            q = [normalise(x) for x in queries[s:s + CHUNK]]
            S = (self.vec.transform(q) @ self.E.T).toarray()
            F = np.maximum.reduceat(S, self.starts, axis=1)
            o = np.argsort(-F, axis=1, kind="stable")[:, :k]
            idx[s:s + len(q)] = o
            sc[s:s + len(q)] = np.take_along_axis(F, o, axis=1)
        return idx, sc

    def exact_match(self, queries: list[str]) -> np.ndarray:
        """Food index when the query is exactly one food's name or alias, else -1 (also -1 when ambiguous)."""
        out = np.full(len(queries), -1, dtype=int)
        for j, q in enumerate(queries):
            hit = self.exact.get(normalise(q))
            if hit and len(hit) == 1:
                out[j] = next(iter(hit))
        return out


def _registry(df: pd.DataFrame, cfg: Config) -> tuple[list[str], list[list[str]]]:
    for c in (cfg.registry_id_col, cfg.registry_name_col, cfg.registry_alias_col):
        if c and c not in df.columns:
            raise MatchingError(f"Registry column not found: {c[:60]}")
    ids = df[cfg.registry_id_col].astype(str).str.strip()
    names = df[cfg.registry_name_col].astype(str).str.strip()
    ok = (ids != "") & (ids.str.lower() != "nan") & (names.map(normalise) != "")
    ids, names = ids[ok], names[ok]
    if ids.duplicated().any():
        raise MatchingError("The registry has repeated food codes. Each food must appear once.")
    if len(ids) < MIN_FOODS:
        raise MatchingError(f"The registry needs at least {MIN_FOODS} foods with a code and a name.")
    base = [[n] for n in names]
    if cfg.registry_alias_col:
        for j, raw in enumerate(df.loc[ok, cfg.registry_alias_col].fillna("").astype(str)):
            base[j].extend(a.strip() for a in ALIAS_SPLIT.split(raw) if a.strip())
    return ids.tolist(), base


def _hits(true: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    return (idx[:, :k] == true[:, None]).any(axis=1)


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(vals)
    m = vals[rng.integers(0, n, size=(BOOTSTRAP, n))].mean(axis=1)
    return [_r(np.percentile(m, 2.5)), _r(np.percentile(m, 97.5))]


def _scores(true: np.ndarray, idx: np.ndarray, ks: list[int]) -> dict:
    return {f"top{k}": _r(_hits(true, idx, k).mean()) for k in ks}


def _fit(ids, base, qn_train: list[str], y_train: np.ndarray) -> Index:
    learned: dict[int, list[str]] = {}
    for q, y in zip(qn_train, y_train):
        learned.setdefault(int(y), []).append(q)
    return Index(ids, base, learned)


def _pick_threshold(score: np.ndarray, correct: np.ndarray, target: float) -> float | None:
    for t in THRESHOLDS:
        m = score >= t
        if m.sum() >= MIN_ANSWERED_FOR_THRESHOLD and correct[m].mean() >= target:
            return t
    return None


def _abstention_table(score: np.ndarray, correct: np.ndarray) -> list[dict]:
    rows = []
    for t in THRESHOLDS:
        m = score >= t
        rows.append({"threshold": t, "answered": int(m.sum()), "coverage": _r(m.mean()),
                     "answered_accuracy": _r(correct[m].mean()) if m.any() else None})
    return rows


def run(pairs: pd.DataFrame, registry: pd.DataFrame, cfg: Config) -> dict:
    warnings: list[str] = []
    for c in (cfg.query_col, cfg.food_col, cfg.group_col):
        if c and c not in pairs.columns:
            raise MatchingError(f"Column not found in the labelled queries: {c[:60]}")
    ids, base = _registry(registry, cfg)
    pos = {f: i for i, f in enumerate(ids)}

    df = pd.DataFrame({"q": pairs[cfg.query_col].astype(str), "f": pairs[cfg.food_col].astype(str).str.strip()})
    if cfg.group_col:
        df["g"] = pairs[cfg.group_col].astype(str).str.strip()
    n_input = len(df)
    df["qn"] = df["q"].map(normalise)
    clean = {"rows_in": n_input}
    bad = (df["qn"] == "") | (df["q"].str.len() > MAX_QUERY_LEN)
    clean["dropped_empty_or_too_long"] = int(bad.sum())
    df = df[~bad]
    unknown = ~df["f"].isin(pos)
    clean["dropped_unknown_food_code"] = int(unknown.sum())
    if unknown.any():
        warnings.append(f"{int(unknown.sum())} labelled queries point to a food code that is not in the registry and were dropped.")
    df = df[~unknown]
    conflict = df.groupby("qn")["f"].transform("nunique") > 1
    clean["dropped_conflicting_labels"] = int(conflict.sum())
    if conflict.any():
        warnings.append(f"{int(conflict.sum())} labelled queries share a written form with a different food and were dropped. "
                        "Resolve these with the data owner: the same words cannot have two correct answers.")
    df = df[~conflict].drop_duplicates(subset=["qn", "f"]).reset_index(drop=True)
    clean["rows_used"] = len(df)
    if len(df) < MIN_PAIRS:
        raise MatchingError(f"At least {MIN_PAIRS} distinct labelled queries are needed after cleaning (found {len(df)}).")
    if df["f"].nunique() < 2:
        raise MatchingError("The labelled queries cover a single food. Matching cannot be evaluated.")
    df["y"] = df["f"].map(pos).astype(int)

    # ----- split by normalised text: one written form never sits on both sides -----
    codes = pd.factorize(df["qn"])[0]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed).split(df, groups=codes))
    if len(te) < MIN_TEST:
        raise MatchingError(f"The hold-out set would have only {len(te)} queries. Add labelled queries or raise test_size.")
    qtr, ytr = df["qn"].iloc[tr].tolist(), df["y"].to_numpy()[tr]
    qte, yte = df["qn"].iloc[te].tolist(), df["y"].to_numpy()[te]
    ks = sorted({1, 3, 5, cfg.top_k})
    ks = [k for k in ks if k <= len(ids)]

    full = _fit(ids, base, qtr, ytr)
    names_only = Index(ids, base)
    f_idx, f_sc = full.top_k(qte, max(ks))
    n_idx, _ = names_only.top_k(qte, max(ks))
    ex = names_only.exact_match(qte)
    rng = np.random.default_rng(cfg.seed)

    f_hit, n_hit = _hits(yte, f_idx, 1), _hits(yte, n_idx, 1)
    diff = f_hit.astype(float) - n_hit.astype(float)
    bi = rng.integers(0, len(diff), size=(BOOTSTRAP, len(diff)))
    dm = diff[bi].mean(axis=1)
    lo, hi = float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))
    verdict = "learning_helps" if lo > 0 else "learning_hurts" if hi < 0 else "no_clear_difference"

    answered = ex >= 0
    exact_block = {"coverage": _r(answered.mean()),
                   "answered_accuracy": _r((ex[answered] == yte[answered]).mean()) if answered.any() else None,
                   "top1_counting_unanswered_as_wrong": _r((ex == yte).mean())}

    # ----- queries that are already a registry name or alias cannot test matching; report them apart -----
    trivial = np.array([normalise(q) in names_only.exact for q in qte])
    nontrivial = {"n": int((~trivial).sum()), "trivial_share": _r(trivial.mean())}
    if (~trivial).sum() >= MIN_TEST:
        nontrivial.update(learned=_scores(yte[~trivial], f_idx[~trivial], ks),
                          names_only=_scores(yte[~trivial], n_idx[~trivial], ks))
    else:
        nontrivial["note"] = "Too few test queries that are not already a registry name to report on them separately."
        warnings.append(nontrivial["note"])
    if trivial.mean() > 0.5:
        warnings.append(f"{round(trivial.mean() * 100)}% of the hold-out queries are exactly a registry name or alias. "
                        "If the queries were generated from the registry, scores overstate real-world performance: "
                        "use queries typed by people, and read the not-already-a-name results.")

    seen = np.isin(yte, ytr)
    unseen_share = _r(1 - seen.mean())
    if unseen_share and unseen_share > 0.3:
        warnings.append(f"{round(unseen_share * 100)}% of hold-out queries are for foods with no labelled query in training. "
                        "For those the matcher can only use the registry names and aliases.")

    # ----- abstention: threshold fixed on training data only, then checked once on the hold-out -----
    oof_s, oof_c = [], []
    ntr_groups = len(set(codes[tr]))
    thr = None
    if ntr_groups >= 9:
        for a, b in GroupKFold(n_splits=3, shuffle=True, random_state=cfg.seed).split(qtr, groups=codes[tr]):
            m = _fit(ids, base, [qtr[i] for i in a], ytr[a])
            i1, s1 = m.top_k([qtr[i] for i in b], 1)
            oof_s.append(s1[:, 0])
            oof_c.append(i1[:, 0] == ytr[b])
        oof_s, oof_c = np.concatenate(oof_s), np.concatenate(oof_c)
        thr = _pick_threshold(oof_s, oof_c, cfg.target_accuracy)
    else:
        warnings.append("Too few training queries to choose a review threshold with cross-validation.")
    top1_correct = f_idx[:, 0] == yte
    abst = {"score_is": "cosine similarity of character n-grams, not a probability", "target_accuracy": cfg.target_accuracy,
            "threshold_chosen_on_training": thr, "holdout_table": _abstention_table(f_sc[:, 0], top1_correct)}
    if thr is not None:
        m = f_sc[:, 0] >= thr
        abst["holdout_at_threshold"] = {"coverage": _r(m.mean()), "answered": int(m.sum()),
                                        "answered_accuracy": _r(top1_correct[m].mean()) if m.any() else None}
        if m.any() and top1_correct[m].mean() < cfg.target_accuracy - 0.05:
            warnings.append("On the hold-out set the answered accuracy at the chosen threshold is clearly below the target. "
                            "Do not rely on the threshold without a larger, independent check.")
    else:
        warnings.append(f"No threshold reached {round(cfg.target_accuracy * 100)}% accuracy on at least {MIN_ANSWERED_FOR_THRESHOLD} "
                        "answered training queries. Every match would need review.")

    # ----- where it goes wrong, counts only (no query text) -----
    wrong = ~top1_correct
    conf: dict[tuple[int, int], int] = {}
    for t, p in zip(yte[wrong], f_idx[wrong, 0]):
        conf[(int(t), int(p))] = conf.get((int(t), int(p)), 0) + 1
    confused = [{"true_food_code": ids[t], "predicted_food_code": ids[p], "count": c}
                for (t, p), c in sorted(conf.items(), key=lambda kv: -kv[1])[:10]]

    groups = None
    if cfg.group_col:
        g = df["g"].iloc[te].to_numpy()
        groups = []
        for name in sorted(set(g.tolist())):
            m = g == name
            row = {"group": name[:60], "n_test": int(m.sum())}
            if m.sum() >= MIN_GROUP_TEST:
                row.update(learned_top1=_r(f_hit[m].mean()), names_only_top1=_r(n_hit[m].mean()))
            else:
                row["note"] = f"Fewer than {MIN_GROUP_TEST} test queries: not scored."
            groups.append(row)

    if len(te) < 100:
        warnings.append("The hold-out set has fewer than 100 queries, so the scores are imprecise. Read the 95% ranges.")
    if len(ids) > 0 and len(set(ytr)) / len(ids) < 0.3:
        warnings.append("Labelled queries cover under 30% of the registry. Scores say little about the other foods.")

    return {
        "kind": "matching_evaluation",
        "method": "character 2-4-gram TF-IDF similarity over registry names, aliases and (learned matcher only) labelled training queries",
        "n_registry_foods": len(ids), "n_foods_in_queries": int(df["f"].nunique()),
        "cleaning": clean, "n_train": len(tr), "n_test": len(te), "seed": cfg.seed,
        "learned": {**_scores(yte, f_idx, ks), "top1_95_range": _boot_ci(f_hit.astype(float), rng)},
        "names_only": {**_scores(yte, n_idx, ks), "top1_95_range": _boot_ci(n_hit.astype(float), rng)},
        "exact_match_rule": exact_block,
        "learning_effect": {"top1_difference": _r(diff.mean()), "95_range": [_r(lo), _r(hi)], "verdict": verdict,
                            "note": "Paired bootstrap on the same hold-out rows. A lead to confirm on new data, not proof."},
        "not_already_a_registry_name": nontrivial,
        "unseen_food_share_in_test": unseen_share,
        "abstention": abst, "most_confused_foods": confused, "groups": groups,
        "warnings": warnings,
        "note": "Candidates are suggestions for a person to confirm. A similarity score is not the chance that a match is right.",
    }
