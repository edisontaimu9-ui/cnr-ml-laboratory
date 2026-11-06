"""Food-name matching evaluation: honest scoring, guards, and no path to the registry, serving or CNR."""
import ast
import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/mt.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import matching  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}
APP_DIR = Path(__file__).resolve().parent.parent / "app"

# (code, registry label, registry aliases, colloquial names that are NOT in the registry)
FOODS = [
    ("F01", "Maize flour, refined", "ufa woyera", "mgaiwa ya white"), ("F02", "Rice, cooked", "mpunga wophika", "ilise"),
    ("F03", "Beans, dried, boiled", "nyemba", "bins"), ("F04", "Groundnuts, roasted", "mtedza", "pinda"),
    ("F05", "Cassava, boiled", "chinangwa", "kasava"), ("F06", "Sweet potato, boiled", "mbatata", "kandolo"),
    ("F07", "Pumpkin leaves, cooked", "nkhwani", "maungu leaves"), ("F08", "Tilapia, fried", "chambo", "tilapya"),
    ("F09", "Small dried fish", "usipa", "kapenta"), ("F10", "Eggs, boiled", "mazira", "eggi"),
    ("F11", "Tomato, raw", "phwetekere", "tomato fresh"), ("F12", "Onion, raw", "anyezi", "onyoni"),
    ("F13", "Banana, ripe", "nthochi", "bananaz"), ("F14", "Mango, ripe", "mango wakupsa", "manjo"),
    ("F15", "Orange, raw", "lalanje", "oranji"), ("F16", "Milk, whole, fresh", "mkaka", "fresh milk"),
    ("F17", "Cooking oil, vegetable", "mafuta ophikira", "oil ya kuphika"), ("F18", "Sugar, white", "shuga", "suga"),
    ("F19", "Bread, white", "buledi", "bred"), ("F20", "Chicken, roasted", "nkhuku", "kuku"),
    ("F21", "Beef, stewed", "ng'ombe", "nyama ya ng'ombe stew"), ("F22", "Soya pieces, cooked", "soya", "soyapizi"),
    ("F23", "Okra, boiled", "therere", "okura"), ("F24", "Cabbage, boiled", "kabichi", "kabbage"),
    ("F25", "Irish potato, boiled", "mbatata ya chizungu", "potatoz"), ("F26", "Millet porridge", "phala la mawere", "mawere"),
    ("F27", "Sorghum flour", "ufa wa manyowa", "manyowa"), ("F28", "Pigeon peas, boiled", "nandolo", "nandoro"),
    ("F29", "Cowpeas leaves", "chisoso", "chisozo"), ("F30", "Honey", "uchi", "huni"),
]


def registry_csv():
    rows = ["food_code,food_label,local_names"]
    rows += [f'{c},"{n}",{a}' for c, n, a, _ in FOODS]
    return "\n".join(rows) + "\n"


def variants(word, rng, n):
    out = {word}
    letters = "abcdefghijklmnopqrstuvwxyz"
    guard = 0
    while len(out) < n and guard < 200:
        guard += 1
        w, i = list(word), int(rng.integers(0, len(word)))
        op = int(rng.integers(0, 3))
        if op == 0 and len(w) > 3:
            del w[i]
        elif op == 1:
            w.insert(i, letters[int(rng.integers(0, 26))])
        else:
            w[i] = letters[int(rng.integers(0, 26))]
        out.add("".join(w))
    return sorted(out)


def queries_csv(seed=5, per=8, typed_only=True):
    """People write colloquial names the registry does not list, often misspelt. Language is a grouping column."""
    rng = np.random.default_rng(seed)
    rows = ["search_text,food_code,language"]
    for c, n, a, coll in FOODS:
        for v in variants(coll, rng, per):
            rows.append(f'"{v}",{c},{"chichewa" if len(v) % 2 else "english"}')
        if not typed_only:
            rows.append(f'"{n}",{c},english')
    return "\n".join(rows) + "\n"


def frames(**kw):
    import io
    return pd.read_csv(io.StringIO(queries_csv(**kw))), pd.read_csv(io.StringIO(registry_csv()))


def cfg(**kw):
    base = dict(query_col="search_text", food_col="food_code", registry_id_col="food_code",
                registry_name_col="food_label", registry_alias_col="local_names")
    base.update(kw)
    return matching.Config(**base)


def test_normalise_folds_case_accents_and_punctuation():
    assert matching.normalise("  Ng'ombe,  STEWED! ") == "ng ombe stewed"
    assert matching.normalise("Café") == "cafe"


def test_learning_from_labelled_queries_beats_names_only_on_colloquial_names():
    q, r = frames()
    res = matching.run(q, r, cfg())
    assert res["learned"]["top1"] > res["names_only"]["top1"] + 0.1
    assert res["learning_effect"]["verdict"] == "learning_helps"
    assert res["learned"]["top5"] >= res["learned"]["top1"]
    assert res["note"].startswith("Candidates are suggestions")


def test_one_written_form_never_sits_in_train_and_test():
    q, r = frames()
    q = pd.concat([q, q, q])          # exact duplicates must collapse, not leak across the split
    res = matching.run(q, r, cfg())
    assert res["cleaning"]["rows_used"] == len(frames()[0].drop_duplicates(subset=["search_text", "food_code"]))
    assert res["n_train"] + res["n_test"] == res["cleaning"]["rows_used"]


def test_registry_generated_queries_are_flagged_as_trivial():
    _, r = frames()
    names = pd.DataFrame({"search_text": r["food_label"], "food_code": r["food_code"], "language": "english"})
    aliases = pd.DataFrame({"search_text": r["local_names"], "food_code": r["food_code"], "language": "chichewa"})
    res = matching.run(pd.concat([names, aliases]), r, cfg(test_size=0.4))      # a dataset made only of registry names
    assert res["not_already_a_registry_name"]["trivial_share"] > 0.9
    assert any("exactly a registry name" in w for w in res["warnings"])


def test_conflicting_labels_and_unknown_codes_are_dropped_and_reported():
    q, r = frames()
    extra = pd.DataFrame({"search_text": ["pinda", "pinda", "zzz"], "food_code": ["F04", "F01", "NOPE"], "language": "english"})
    res = matching.run(pd.concat([q, extra]), r, cfg())
    assert res["cleaning"]["dropped_conflicting_labels"] >= 2
    assert res["cleaning"]["dropped_unknown_food_code"] == 1
    assert any("different food" in w for w in res["warnings"])


def test_abstention_threshold_is_chosen_on_training_and_scores_are_not_called_probabilities():
    q, r = frames(per=14)
    res = matching.run(q, r, cfg(target_accuracy=0.9))
    a = res["abstention"]
    assert "not a probability" in a["score_is"]
    assert a["threshold_chosen_on_training"] is not None
    cov = [row["coverage"] for row in a["holdout_table"]]
    assert cov == sorted(cov, reverse=True)
    assert "holdout_at_threshold" in a


def test_group_breakdown_and_no_query_text_in_results():
    q, r = frames(per=14)
    res = matching.run(q, r, cfg(group_col="language"))
    assert {g["group"] for g in res["groups"]} == {"chichewa", "english"}
    blob = str(res)
    for word in ("kapenta", "pinda", "kasava"):
        assert word not in blob


def test_guards():
    q, r = frames()
    with pytest.raises(matching.MatchingError, match="At least 50"):
        matching.run(q.head(30), r, cfg())
    with pytest.raises(matching.MatchingError, match="at least 20 foods"):
        matching.run(q, r.head(10), cfg())
    with pytest.raises(matching.MatchingError, match="repeated food codes"):
        matching.run(q, pd.concat([r, r.head(1)]), cfg())
    with pytest.raises(matching.MatchingError, match="not found"):
        matching.run(q, r, cfg(query_col="nope"))
    one = q[q["food_code"] == "F01"]
    with pytest.raises(matching.MatchingError):
        matching.run(pd.concat([one] * 8).assign(search_text=lambda d: [f"x{i}" for i in range(len(d))]), r, cfg())


def test_same_seed_same_result():
    q, r = frames()
    assert matching.run(q, r, cfg()) == matching.run(q, r, cfg())


# ---------------- API ----------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def upload(client, name, csv):
    d = client.post("/v1/datasets", json={"name": name}, headers=ADMIN).json()
    r = client.post(f"/v1/datasets/{d['id']}/versions", headers=ADMIN, data={"approval_ref": "TEST-MATCH"},
                    files={"file": ("m.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    return d["id"]


BODY = dict(query_column="search_text", food_code_column="food_code", registry_code_column="food_code",
            registry_label_column="food_label", registry_aliases_column="local_names")


def test_api_evaluates_and_is_not_deployable(client):
    qid, rid = upload(client, "match-queries", queries_csv()), upload(client, "match-registry", registry_csv())
    r = client.post("/v1/matching/evaluations", headers=ADMIN, json={**BODY, "queries_dataset_id": qid, "registry_dataset_id": rid})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["deployable"] is False and j["kind"] == "matching_evaluation"
    assert j["results"]["learning_effect"]["verdict"] == "learning_helps"
    log = client.get("/v1/audit", headers=ADMIN).json()
    assert any(e["action"] == "matching.evaluated" for e in (log if isinstance(log, list) else log.get("items", [])))


def test_api_unknown_column_dataset_and_permissions(client):
    qid, rid = upload(client, "match-queries-2", queries_csv()), upload(client, "match-registry-2", registry_csv())
    ok = {**BODY, "queries_dataset_id": qid, "registry_dataset_id": rid}
    assert client.post("/v1/matching/evaluations", headers=ADMIN, json={**ok, "query_column": "nope"}).status_code == 422
    assert client.post("/v1/matching/evaluations", headers=ADMIN, json={**ok, "registry_dataset_id": 9999}).status_code == 404
    assert client.post("/v1/matching/evaluations", json=ok).status_code in (401, 403)
    key = client.post("/v1/admin/keys", headers=ADMIN, json={"name": "viewer key", "role": "viewer"}).json()["key"]
    assert client.post("/v1/matching/evaluations", headers={"Authorization": "Bearer " + key}, json=ok).status_code == 403


def test_matching_is_invisible_to_the_registry_and_serving():
    banned = {"registry", "serving", "predict", "webhook", "integration"}
    for f in ("matching.py", "routers/matching.py"):
        tree = ast.parse((APP_DIR / f).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [(node.module or "")] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                assert not (set(n.replace("..", ".").split(".")) & banned), f"{f} imports {n}"
        assert "joblib" not in (APP_DIR / f).read_text() and "save_bytes" not in (APP_DIR / f).read_text()
