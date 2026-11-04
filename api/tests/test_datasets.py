import io
import json
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/ds.db")
os.environ.setdefault("BOOTSTRAP_ADMIN_KEY", "x" * 40)
os.environ.setdefault("ENVIRONMENT", "test")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import storage  # noqa: E402
from app.main import app  # noqa: E402

ADMIN = {"Authorization": "Bearer " + os.environ["BOOTSTRAP_ADMIN_KEY"]}

CSV = (
    "Patient Name,Phone,Sex,District,Date of Birth,Age Months,Weight kg,Notes,Hospital Number\n"
    + "\n".join(
        f"Person {i},0999{100000 + i},{'F' if i % 2 else 'M'},{'Zomba' if i % 3 else 'Blantyre'},"
          f"{2018 + i % 4}-0{1 + i % 9}-1{i % 9},{10 + i % 30},{8.5 + (i % 40) / 10},ok,HN{i:05d}"
        for i in range(60)
    )
    + "\n"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ds(client):
    r = client.post("/v1/datasets", json={"name": "pilot", "description": "t"}, headers=ADMIN)
    assert r.status_code == 201
    return r.json()


def upload(client, path, data, name="d.csv", **form):
    return client.post(path, headers=ADMIN, files={"file": (name, data, "text/csv")}, data=form)


def test_inspect_flags_identifiers_and_hides_their_values(client):
    r = upload(client, "/v1/datasets/inspect", CSV)
    assert r.status_code == 200
    body = r.json()
    cols = {c["name"]: c for c in body["columns"]}
    assert cols["patient_name"]["category"] == "person_name" and cols["patient_name"]["suggested"] == "drop"
    assert cols["phone"]["category"] in ("contact", "value_pii") and not cols["phone"]["can_keep"] or cols["phone"]["category"] == "contact"
    assert cols["hospital_number"]["suggested"] == "drop"
    assert cols["date_of_birth"]["suggested"] == "year_only"
    assert cols["weight_kg"]["kind"] == "numeric"
    # PII values must never be echoed back
    assert "Person 1" not in json.dumps(body) and "0999" not in json.dumps(body)
    assert cols["patient_name"]["profile"].get("hidden") is True


def test_create_version_applies_defaults_and_stores_only_deidentified(client, ds):
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", CSV, approval_ref="MREC-2026-001")
    assert r.status_code == 201, r.text
    v = r.json()
    assert v["version"] == 1 and v["deidentified"] is True and v["row_count"] == 60
    names = [c["name"] for c in v["schema_info"]["columns"]]
    for gone in ("patient_name", "phone", "hospital_number", "notes"):
        assert gone not in names or gone == "notes"
    assert "patient_name" not in names and "phone" not in names and "hospital_number" not in names
    assert v["report_json"]["generalised"] == [{"column": "date_of_birth", "to": "year"}]
    # stored file contains no identifiers
    stored = storage.load(f"local://ds{ds['id']}/v1.csv.gz")
    assert "Person 1" not in stored.to_csv() and "0999" not in stored.to_csv()
    assert stored["date_of_birth"].between(2018, 2021).all()


def test_duplicate_file_rejected(client, ds):
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", CSV, approval_ref="MREC-2026-001")
    assert r.status_code == 409


def test_keeping_flagged_column_needs_reason(client, ds):
    data = CSV + "Extra,0999000000,F,Zomba,2019-01-01,12,9.1,ok,HN99999\n"
    plan = {"columns": [{"name": "patient_name", "action": "keep"}]}
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", data, approval_ref="MREC-2026-001", plan=json.dumps(plan))
    assert r.status_code == 422 and "written reason" in r.json()["detail"]
    plan = {"columns": [{"name": "patient_name", "action": "keep", "override_reason": "needed for twin matching"}]}
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", data, approval_ref="MREC-2026-001", plan=json.dumps(plan))
    assert r.status_code == 201 and r.json()["report_json"]["overrides"][0]["column"] == "patient_name"


def test_value_pii_can_never_be_kept(client, ds):
    data = "id_x,comment,wt\n" + "\n".join(f"{i},mail me a{i}@example.org,{i}" for i in range(30)) + "\n"
    plan = {"columns": [{"name": "comment", "action": "keep", "override_reason": "please keep it anyway"}]}
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", data, approval_ref="MREC-2026-002", plan=json.dumps(plan))
    assert r.status_code == 422 and "cannot be kept" in r.json()["detail"]


def test_pseudonymise_is_stable_and_not_the_original(client, ds):
    data = "child_id,wt\n" + "\n".join(f"C{i % 5},{i}" for i in range(30)) + "\n"
    plan = {"columns": [{"name": "child_id", "action": "pseudonymise"}]}
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", data, approval_ref="MREC-2026-003", plan=json.dumps(plan))
    assert r.status_code == 201, r.text
    v = r.json()
    stored = storage.load(f"local://ds{ds['id']}/v{v['version']}.csv.gz")
    assert stored["child_id"].nunique() == 5 and not stored["child_id"].str.startswith("C").any()


def test_bad_files_rejected(client, ds):
    assert upload(client, "/v1/datasets/inspect", b"a,b\n", name="x.csv").status_code == 422
    assert upload(client, "/v1/datasets/inspect", b"a,b\n1,2\n", name="x.exe").status_code == 422
    assert upload(client, "/v1/datasets/inspect", b"PK\x03\x04junk", name="x.csv").status_code == 422
    assert upload(client, "/v1/datasets/inspect", b"junk", name="x.xlsx").status_code == 422
    r = upload(client, f"/v1/datasets/{ds['id']}/versions", CSV + "z\n", approval_ref="x")
    assert r.status_code == 422  # approval reference too short


def test_xlsx_supported(client, ds):
    buf = io.BytesIO()
    pd.DataFrame({"age": [1, 2, 3, 4], "sex": ["F", "M", "F", "M"]}).to_excel(buf, index=False)
    r = client.post(f"/v1/datasets/{ds['id']}/versions", headers=ADMIN,
                    files={"file": ("t.xlsx", buf.getvalue())}, data={"approval_ref": "MREC-2026-004"})
    assert r.status_code == 201 and r.json()["file_format"] == "xlsx"


def test_permissions_and_preview(client, ds):
    viewer = client.post("/v1/admin/keys", json={"name": "v1", "role": "viewer"}, headers=ADMIN).json()["key"]
    h = {"X-API-Key": viewer}
    assert client.get("/v1/datasets", headers=h).status_code == 200
    assert client.post("/v1/datasets", json={"name": "nope"}, headers=h).status_code == 403
    assert client.get(f"/v1/datasets/{ds['id']}/versions/1/preview", headers=h).status_code == 403
    r = client.get(f"/v1/datasets/{ds['id']}/versions/1/preview", headers=ADMIN)
    assert r.status_code == 200 and len(r.json()["rows"]) == 20
    assert "dataset.preview_viewed" in client.get("/v1/audit", headers=ADMIN).text


def test_audit_has_no_row_content(client):
    log = client.get("/v1/audit?limit=200", headers=ADMIN).text
    assert "dataset.version_created" in log and "Person 1" not in log and "0999" not in log


def test_duplicate_dataset_name(client, ds):
    assert client.post("/v1/datasets", json={"name": "pilot"}, headers=ADMIN).status_code == 409


def test_unhandled_error_returns_json_with_cors(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(storage, "save", boom)
    c = TestClient(app, raise_server_exceptions=False)
    d = c.post("/v1/datasets", json={"name": "boom-ds"}, headers=ADMIN).json()
    r = c.post(f"/v1/datasets/{d['id']}/versions", headers={**ADMIN, "Origin": "http://localhost:8080"},
               files={"file": ("d.csv", "a,b\n1,2\n3,4\n5,6\n")}, data={"approval_ref": "TEST-001"})
    assert r.status_code == 500 and "Server error" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8080"
