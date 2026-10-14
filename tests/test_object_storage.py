from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from config.object_storage import ObjectStore, ObjectStoreError
import app as dashboard


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.fail_upload = False
        self.fail_put = False
        self.fail_after_put = None
        self.put_count = 0

    def get_paginator(self, _operation):
        return self

    def paginate(self, *, Bucket, Prefix):
        contents = [
            {"Key": key, "ETag": '"' + hashlib.md5(data).hexdigest() + '"', "Size": len(data)}
            for (bucket, key), data in self.objects.items()
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return [{"Contents": contents}]

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, *, Bucket, Key, Body, Metadata=None, **_kwargs):
        if self.fail_put:
            raise RuntimeError("simulated put failure")
        if self.fail_after_put is not None and self.put_count >= self.fail_after_put:
            raise RuntimeError("simulated interrupted upload")
        self.put_count += 1
        self.objects[(Bucket, Key)] = bytes(Body)
        self.metadata[(Bucket, Key)] = Metadata or {}

    def upload_file(self, filename, bucket, key, ExtraArgs=None, Callback=None, Config=None):
        if self.fail_upload:
            raise RuntimeError("simulated upload failure")
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def head_object(self, *, Bucket, Key):
        data = self.objects.get((Bucket, Key))
        if data is None:
            raise MissingObject()
        return {"ETag": '"' + hashlib.md5(data).hexdigest() + '"',
                "ContentLength": len(data),
                "Metadata": self.metadata.get((Bucket, Key), {})}

    def copy_object(self, *, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.metadata[(Bucket, Key)] = self.metadata.get((CopySource["Bucket"], CopySource["Key"]), {})


class MissingObject(Exception):
    response = {"Error": {"Code": "404"}}


def test_file_is_available_to_a_fresh_instance_after_request(tmp_path):
    remote = MemoryS3()
    first_root = tmp_path / "instance-a"
    (first_root / "results").mkdir(parents=True)
    artifact = first_root / "results" / "map.tif"
    artifact.write_bytes(b"persisted raster content")
    ObjectStore("bucket", remote, first_root).upload_roots(["results"])

    new_root = tmp_path / "instance-b"
    restored = ObjectStore("bucket", remote, new_root)
    assert restored.sync_roots(["results"]) == 1
    assert (new_root / "results" / "map.tif").read_bytes() == b"persisted raster content"


def test_changed_artifact_keeps_previous_version(tmp_path):
    remote = MemoryS3()
    root = tmp_path / "app"
    output = root / "results" / "report.json"
    output.parent.mkdir(parents=True)
    output.write_text("old")
    store = ObjectStore("bucket", remote, root)
    store.upload_roots(["results"])
    output.write_text("new")

    assert store.upload_roots(["results"]) == 1
    assert remote.objects[("bucket", "results/report.json")] == b"new"
    assert remote.objects[("bucket", "results/report.json")] != b"old"
    assert any(key.startswith("history/") and value == b"old"
               for (bucket, key), value in remote.objects.items())


def test_refuses_to_overwrite_a_concurrent_instance_change(tmp_path):
    remote = MemoryS3()
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    (root_a / "results").mkdir(parents=True)
    (root_a / "results" / "out.json").write_text("initial")
    store_a = ObjectStore("bucket", remote, root_a)
    store_a.upload_roots(["results"])
    store_b = ObjectStore("bucket", remote, root_b)
    store_b.sync_roots(["results"])

    (root_a / "results" / "out.json").write_text("writer A")
    store_a.upload_roots(["results"])
    (root_b / "results" / "out.json").write_text("writer B")
    with pytest.raises(ObjectStoreError, match="changed on another instance"):
        store_b.upload_roots(["results"])


def test_failed_upload_is_reported(tmp_path):
    remote = MemoryS3()
    root = tmp_path / "app"
    (root / "results").mkdir(parents=True)
    (root / "results" / "out.json").write_text("data")
    remote.fail_upload = True

    with pytest.raises(ObjectStoreError, match="could not save persistent file"):
        ObjectStore("bucket", remote, root).upload_roots(["results"])


@pytest.mark.parametrize("key", ["results/../../secret", "logs/app.log", "data/cache/tile.bin"])
def test_rejects_unsafe_or_temporary_paths(tmp_path, key):
    store = ObjectStore("bucket", MemoryS3(), tmp_path)
    with pytest.raises(ObjectStoreError):
        store.sync_keys([key])


def test_never_uploads_smoke_model_or_log_files(tmp_path):
    remote = MemoryS3()
    root = tmp_path / "app"
    (root / "models").mkdir(parents=True)
    (root / "models" / "flood_unet_smoke.pt").write_bytes(b"test weights")
    (root / "data").mkdir()
    (root / "data" / "download.log").write_text("temporary log")

    assert ObjectStore("bucket", remote, root).upload_roots(["models", "data"]) == 0
    assert not remote.objects


def test_large_file_uses_verified_chunks_and_restores_on_new_instance(tmp_path):
    remote = MemoryS3()
    first_root = tmp_path / "large-source"
    source = first_root / "data" / "large-scene.zip"
    source.parent.mkdir(parents=True)
    with source.open("wb") as stream:
        for value in range(5):
            stream.write(bytes([value]) * (8 * 1024 * 1024))
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    assert ObjectStore("bucket", remote, first_root).upload_roots(["data"]) == 1
    assert remote.metadata[("bucket", "data/large-scene.zip")]["jalnetra-chunked"] == "1"
    second_root = tmp_path / "large-instance"
    assert ObjectStore("bucket", remote, second_root).sync_roots(["data"]) == 1
    restored = second_root / "data" / "large-scene.zip"
    assert hashlib.sha256(restored.read_bytes()).hexdigest() == expected
    assert not list((second_root / "data").rglob("*.__parts__*"))


def test_interrupted_chunk_upload_resumes_saved_parts(tmp_path):
    remote = MemoryS3()
    root = tmp_path / "source"
    source = root / "data" / "large-scene.zip"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"scene" * (9 * 1024 * 1024))
    remote.fail_after_put = 2

    with pytest.raises(ObjectStoreError, match="RuntimeError"):
        ObjectStore("bucket", remote, root).upload_roots(["data"])

    saved_parts = remote.put_count
    remote.fail_after_put = None
    assert ObjectStore("bucket", remote, root).upload_roots(["data"]) == 1
    assert remote.put_count == saved_parts + 5
    assert remote.metadata[("bucket", "data/large-scene.zip")]["jalnetra-chunked"] == "1"


def test_dashboard_report_is_restored_from_a_fresh_instance(tmp_path, monkeypatch):
    remote = MemoryS3()
    source = tmp_path / "source"
    report = source / "report" / "situation_report.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html><body><div class=\"page\">saved report</div></body></html>")
    ObjectStore("bucket", remote, source).upload_roots(["report"])

    instance = tmp_path / "instance"
    monkeypatch.setattr(dashboard, "OBJECT_STORE", ObjectStore("bucket", remote, instance))
    monkeypatch.setattr(dashboard, "RESULTS", instance / "results")
    monkeypatch.setattr(dashboard, "REPORT_DIR", instance / "report")
    monkeypatch.setattr(dashboard, "LOGIN_REQUIRED", False)
    response = dashboard.app.test_client().get("/report")

    assert response.status_code == 200
    assert "saved report" in response.get_data(as_text=True)


def test_unauthorized_dashboard_request_does_not_read_storage(monkeypatch):
    class DenyStorage:
        def sync_roots(self, _roots):
            raise AssertionError("storage should not be reached before authentication")

    monkeypatch.setattr(dashboard, "OBJECT_STORE", DenyStorage())
    monkeypatch.setattr(dashboard, "LOGIN_REQUIRED", True)
    monkeypatch.setattr(dashboard, "PIPELINE_TOKEN", "configured-token")
    response = dashboard.app.test_client().get("/api/status")

    assert response.status_code == 401
