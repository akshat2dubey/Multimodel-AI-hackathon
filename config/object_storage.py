"""S3-compatible persistence for files shared by the dashboard and pipeline."""

from __future__ import annotations

import json
import hashlib
import mimetypes
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PERSISTENT_ROOTS = ("data", "data_demo", "results", "results_demo", "report", "models")
CHUNK_THRESHOLD = 32 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _serialized(method):
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class ObjectStoreError(RuntimeError):
    """An external object-store read or write failed."""


class ObjectStore:
    def __init__(self, bucket: str, client, root: Path, progress_callback=None,
                 transfer_config=None):
        self.bucket, self.client, self.root = bucket, client, root.resolve()
        self.state_path = self.root / ".antideploy-sync-state.json"
        self._lock = threading.RLock()
        self.progress_callback = progress_callback
        self.transfer_config = transfer_config

    @classmethod
    def from_env(cls, root: Path, *, required: bool = False):
        bucket = os.getenv("BUCKET_NAME") or os.getenv("S3_BUCKET") or os.getenv("AWS_S3_BUCKET")
        endpoint = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("S3_ENDPOINT")
        region = os.getenv("AWS_REGION") or os.getenv("S3_REGION")
        access = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        values = (bucket, endpoint, region, access, secret)
        if not any(values):
            if required:
                raise ObjectStoreError("persistent object storage is required but not configured")
            return None
        if not all(values):
            raise ObjectStoreError("persistent object storage configuration is incomplete")
        try:
            import boto3
            from botocore.config import Config
            from boto3.s3.transfer import TransferConfig

            client = boto3.client(
                "s3", endpoint_url=endpoint, region_name=region,
                aws_access_key_id=access, aws_secret_access_key=secret,
                config=Config(s3={"addressing_style": "path"},
                              connect_timeout=10, read_timeout=300,
                              retries={"max_attempts": 4, "mode": "standard"}),
            )
            transfer_config = TransferConfig(
                multipart_threshold=1024 * 1024 * 1024,
                multipart_chunksize=64 * 1024 * 1024,
                max_concurrency=1,
            )
        except Exception as exc:
            raise ObjectStoreError("could not initialize persistent object storage") from exc
        return cls(bucket, client, root, transfer_config=transfer_config)

    @staticmethod
    def _ephemeral(key: str) -> bool:
        parts = PurePosixPath(key).parts
        name = parts[-1].lower() if parts else ""
        return (
            "cache" in parts or name.startswith(".")
            or name.endswith((".log", ".tmp"))
            or any(".__parts__" in part for part in parts)
            or name == "flood_unet_smoke.pt" or "_smoke." in name
            or key == "results/flood_mask.geojson"
        )

    @staticmethod
    def _valid_key(key: str) -> bool:
        path = PurePosixPath(key)
        return (not path.is_absolute() and bool(path.parts) and ".." not in path.parts
                and path.parts[0] in PERSISTENT_ROOTS and not ObjectStore._ephemeral(key))

    def _path(self, key: str) -> Path:
        if not self._valid_key(key):
            raise ObjectStoreError("refusing an unsafe or non-persistent object path")
        path = self.root.joinpath(*PurePosixPath(key).parts)
        base = self.root / PurePosixPath(key).parts[0]
        try:
            base.resolve().relative_to(self.root)
            path.resolve().relative_to(base.resolve())
        except ValueError as exc:
            raise ObjectStoreError("refusing a path outside the project directory") from exc
        return path

    def _read_state(self) -> dict:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("_bucket") == self.bucket:
                return value
            return {"_bucket": self.bucket}
        except (OSError, ValueError):
            return {"_bucket": self.bucket}

    def _write_state(self, state: dict) -> None:
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self.state_path)

    def _download_object(self, key: str, head: dict, filename: str) -> tuple[int, str | None]:
        metadata = head.get("Metadata", {})
        if metadata.get("jalnetra-chunked") != "1":
            self.client.download_file(self.bucket, key, filename)
            return int(head.get("ContentLength", 0)), None

        try:
            size = int(metadata["jalnetra-size"])
            count = int(metadata["jalnetra-parts"])
            prefix = metadata["jalnetra-prefix"]
            checksum = metadata["jalnetra-sha256"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ObjectStoreError(f"invalid chunk manifest for {key}") from exc
        if (size < CHUNK_THRESHOLD or count != (size + CHUNK_BYTES - 1) // CHUNK_BYTES
                or prefix.rsplit("/", 1)[0] != f"{key}.__parts__"):
            raise ObjectStoreError(f"invalid chunk manifest for {key}")

        digest = hashlib.sha256()
        written = 0
        with open(filename, "wb") as output:
            for part in range(count):
                part_key = f"{prefix}/{part:06d}"
                try:
                    response = self.client.get_object(Bucket=self.bucket, Key=part_key)
                    body = response["Body"]
                    while block := body.read(1024 * 1024):
                        output.write(block)
                        digest.update(block)
                        written += len(block)
                    body.close()
                except Exception as exc:
                    raise ObjectStoreError(f"could not retrieve persistent file {key}") from exc
        if written != size or digest.hexdigest() != checksum:
            raise ObjectStoreError(f"integrity check failed for persistent file {key}")
        return size, checksum

    def _upload_chunked(self, path: Path, key: str, content_type: str) -> None:
        # A content-addressed prefix lets retries continue after a transient
        # network failure without retransmitting parts that were already saved.
        file_digest = _sha256(path).hex()
        prefix = f"{key}.__parts__/{file_digest}"
        digest = hashlib.sha256()
        parts = 0
        try:
            with path.open("rb") as stream:
                while block := stream.read(CHUNK_BYTES):
                    digest.update(block)
                    part_key = f"{prefix}/{parts:06d}"
                    part_digest = hashlib.sha256(block).hexdigest()
                    try:
                        existing = self.client.head_object(Bucket=self.bucket, Key=part_key)
                    except Exception as exc:
                        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
                        if code not in ("404", "NoSuchKey", "NotFound"):
                            raise
                        existing = None
                    metadata = (existing or {}).get("Metadata", {})
                    if (not existing or metadata.get("jalnetra-part-sha256") != part_digest
                            or int(existing.get("ContentLength", -1)) != len(block)):
                        self.client.put_object(
                            Bucket=self.bucket, Key=part_key,
                            Body=block, ContentLength=len(block),
                            ContentType="application/octet-stream",
                            Metadata={"jalnetra-part-sha256": part_digest},
                        )
                        if self.progress_callback:
                            self.progress_callback(len(block))
                    parts += 1
            manifest = {
                "format": "jalnetra-chunked-v1",
                "size": path.stat().st_size,
                "parts": parts,
                "sha256": digest.hexdigest(),
            }
            payload = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=payload, ContentLength=len(payload),
                ContentType=content_type,
                Metadata={
                    "jalnetra-chunked": "1",
                    "jalnetra-size": str(path.stat().st_size),
                    "jalnetra-parts": str(parts),
                    "jalnetra-prefix": prefix,
                    "jalnetra-sha256": digest.hexdigest(),
                },
            )
        except Exception as exc:
            if isinstance(exc, ObjectStoreError):
                raise
            response = getattr(exc, "response", None) or {}
            detail = response.get("Error", {}).get("Code") or type(exc).__name__
            raise ObjectStoreError(f"could not save persistent file {key} (storage error {detail})") from exc

    def _list(self, roots: set[str]) -> dict[str, dict]:
        result = {}
        try:
            for root in roots:
                pages = self.client.get_paginator("list_objects_v2").paginate(
                    Bucket=self.bucket, Prefix=f"{root}/"
                )
                for page in pages:
                    for item in page.get("Contents", []):
                        key = item["Key"]
                        if self._valid_key(key):
                            result[key] = item
        except Exception as exc:
            raise ObjectStoreError("could not list persistent pipeline files") from exc
        return result

    def _restore(self, key: str, remote: dict, state: dict) -> bool:
        path = self._path(key)
        etag = str(remote.get("ETag", "")).strip('"')
        known = state.get(key, {})
        try:
            stat = path.stat()
            if (path.is_file() and known.get("etag") == etag
                    and known.get("size") == stat.st_size
                    and known.get("mtime_ns") == stat.st_mtime_ns):
                return False
            if path.is_file() and known and stat.st_mtime_ns != known.get("mtime_ns"):
                raise ObjectStoreError(f"local persistent file has uncommitted changes: {key}")
        except OSError:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".restore-", dir=path.parent)
        os.close(fd)
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
            if str(head.get("ETag", "")).strip('"') != etag:
                raise ObjectStoreError(f"persistent file changed during restore: {key}")
            size, _ = self._download_object(key, head, tmp_name)
            if Path(tmp_name).stat().st_size != size:
                raise ObjectStoreError(f"download size check failed for {key}")
            if path.is_file() and not known:
                if _sha256(path) != _sha256(Path(tmp_name)):
                    raise ObjectStoreError(f"refusing to replace an untracked local file: {key}")
                state[key] = {"etag": etag, "size": size, "mtime_ns": path.stat().st_mtime_ns}
                return False
            os.replace(tmp_name, path)
        except Exception as exc:
            Path(tmp_name).unlink(missing_ok=True)
            if isinstance(exc, ObjectStoreError):
                raise
            raise ObjectStoreError(f"could not retrieve persistent file {key}") from exc
        state[key] = {"etag": etag, "size": size, "mtime_ns": path.stat().st_mtime_ns}
        return True

    @_serialized
    def sync_keys(self, keys: list[str]) -> int:
        if any(not self._valid_key(key) for key in keys):
            raise ObjectStoreError("refusing an unsafe or non-persistent object path")
        roots = {key.split("/", 1)[0] for key in keys}
        objects = self._list(roots)
        state = self._read_state()
        count = 0
        for key in keys:
            remote = objects.get(key)
            if remote:
                count += self._restore(key, remote, state)
        self._write_state(state)
        return count

    @_serialized
    def sync_roots(self, roots: list[str]) -> int:
        if any(root not in PERSISTENT_ROOTS for root in roots):
            raise ObjectStoreError("refusing an unknown persistence directory")
        objects = self._list(set(roots))
        state = self._read_state()
        count = 0
        for key, remote in objects.items():
            count += self._restore(key, remote, state)
        self._write_state(state)
        return count

    @_serialized
    def upload_file(self, key: str, *, preserve_previous: bool = False) -> None:
        path = self._path(key)
        if not path.is_file():
            raise ObjectStoreError(f"persistent file is missing: {key}")
        state = self._read_state()
        previous = state.get(key, {})
        try:
            remote = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code not in ("404", "NoSuchKey", "NotFound"):
                raise ObjectStoreError(f"could not verify persistent file {key}") from exc
            remote = None
        if remote and previous and previous.get("etag") != str(remote.get("ETag", "")).strip('"'):
            raise ObjectStoreError(f"persistent file changed on another instance: {key}")
        if remote and not previous:
            fd, tmp_name = tempfile.mkstemp(prefix=".verify-", dir=path.parent)
            os.close(fd)
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
                self._download_object(key, head, tmp_name)
                if (Path(tmp_name).stat().st_size != path.stat().st_size
                        or _sha256(path) != _sha256(Path(tmp_name))):
                    raise ObjectStoreError(
                        f"refusing to overwrite an untracked persistent file: {key}"
                    )
            except ObjectStoreError:
                raise
            except Exception as exc:
                raise ObjectStoreError(f"could not verify existing persistent file {key}") from exc
            finally:
                Path(tmp_name).unlink(missing_ok=True)
            state[key] = {"etag": str(remote.get("ETag", "")).strip('"'),
                          "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            self._write_state(state)
            return
        if remote and preserve_previous:
            version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
            try:
                self.client.copy_object(
                    Bucket=self.bucket, Key=f"history/{version}/{key}",
                    CopySource={"Bucket": self.bucket, "Key": key},
                )
            except Exception as exc:
                raise ObjectStoreError(f"could not preserve the previous version of {key}") from exc
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            if path.stat().st_size > CHUNK_THRESHOLD:
                self._upload_chunked(path, key, content_type)
            else:
                self.client.upload_file(
                    str(path), self.bucket, key,
                    ExtraArgs={"ContentType": content_type},
                    Callback=self.progress_callback,
                    Config=self.transfer_config,
                )
            current = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if isinstance(exc, ObjectStoreError):
                raise
            response = getattr(exc, "response", None) or {}
            detail = response.get("Error", {}).get("Code") or response.get(
                "ResponseMetadata", {}
            ).get("HTTPStatusCode") or type(exc).__name__
            raise ObjectStoreError(
                f"could not save persistent file {key} (storage error {detail})"
            ) from exc
        state[key] = {
            "etag": str(current.get("ETag", "")).strip('"'),
            "size": int(current.get("Metadata", {}).get(
                "jalnetra-size", current.get("ContentLength", path.stat().st_size)
            )),
            "mtime_ns": path.stat().st_mtime_ns,
        }
        self._write_state(state)

    @_serialized
    def upload_roots(self, roots: list[str], *, stable_seconds: float = 0) -> int:
        if any(root not in PERSISTENT_ROOTS for root in roots):
            raise ObjectStoreError("refusing an unknown persistence directory")
        remote = self._list(set(roots))
        state = self._read_state()
        uploaded = 0
        for root in roots:
            base = self._path(root)
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                key = path.relative_to(self.root).as_posix()
                if not self._valid_key(key):
                    continue
                try:
                    path.resolve().relative_to(base.resolve())
                except ValueError as exc:
                    raise ObjectStoreError("refusing a symlink outside a persistent directory") from exc
                stat = path.stat()
                if stable_seconds and time.time() - stat.st_mtime < stable_seconds:
                    continue
                known = state.get(key, {})
                if (known and known.get("size") == stat.st_size
                        and known.get("mtime_ns") == stat.st_mtime_ns
                        and known.get("etag") == str(remote.get(key, {}).get("ETag", "")).strip('"')):
                    continue
                self.upload_file(key, preserve_previous=key in remote)
                state = self._read_state()
                uploaded += 1
        return uploaded
