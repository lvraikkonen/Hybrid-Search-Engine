"""Local catalog generation and a fail-closed maintenance gate shared by API/CLI."""
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import secrets

from documents import INDEX
from mock_auth import load_profiles
from tools.extract_playbook_sql import canonical_hash

ROOT = Path(__file__).resolve().parent


def atomic_json(path, value):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("x") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class CatalogStore:
    """One local deployment, plus a fixed isolated namespace for live S6 checks."""
    def __init__(self, validation=False, directory=None):
        self.validation = validation
        self.directory = Path(directory) if directory else ROOT / ".local/s6" / ("validation" if validation else "local")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError("Catalog directory cannot be a symlink")
        self.directory.chmod(0o700)
        self.index = "playbook-s6-validation" if validation else INDEX
        self.reader = "pb_s6_reader" if validation else "pb_document_reader"
        self.importer = "pb_s6_importer" if validation else "pb_s2_importer"
        self.credentials = self.directory / "credentials" if validation else ROOT / ".local/s3"
        self.gate = self.directory / "maintenance.json"

    def current(self):
        path = self.directory / "active.json"
        return json.loads(path.read_text()) if path.exists() else None

    def initial_profiles(self):
        profiles = load_profiles()
        if not self.validation:
            return profiles
        result = {}
        for pid in ("pb_msft_01", "pb_partner_nce_01", "pb_partner_both_01"):
            profile = copy.deepcopy(profiles[pid])
            key = pid.replace("pb_", "pb_s6_", 1)
            profile["profile_id"] = key
            profile["authentication_info"].update(UserName=key, UserId="s6-" + key)
            profile["description"] = "S6 隔离验证 · " + profile["description"]
            result[key] = profile
        return result

    def baseline(self, snapshot, tables):
        profiles = self.initial_profiles()
        return {"generation": canonical_hash({"snapshot": snapshot, "profiles": profiles}),
                "snapshot": snapshot, "tables": tables, "profiles": profiles, "batch": None}

    @contextmanager
    def read(self):
        # ponytail: POSIX locks on this one host; multi-host serving needs a
        # transactional shared catalog before this becomes a remote service.
        with (self.directory / "catalog.lock").open("a+") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            try:
                if self.gate.exists():
                    raise RuntimeError("Catalog maintenance in progress")
                yield self.current()
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @contextmanager
    def maintenance(self, plan_id):
        with (self.directory / "writer.lock").open("a+") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            recovering = self.gate.exists()
            if recovering and json.loads(self.gate.read_text())["plan_id"] != plan_id:
                raise RuntimeError("A different failed batch requires recovery first")
            atomic_json(self.gate, {"plan_id": plan_id, "state": "blocked_until_successful_retry"})
            with (self.directory / "catalog.lock").open("a+") as reader:
                fcntl.flock(reader, fcntl.LOCK_EX)
                try:
                    yield recovering
                except BaseException:
                    # Keep the durable marker even after the process crashes.
                    raise
                else:
                    self.gate.unlink()
                finally:
                    fcntl.flock(reader, fcntl.LOCK_UN)


@contextmanager
def legacy_setup():
    """S2/S3 bootstrap cannot race with or undo a committed S6 policy."""
    store = CatalogStore()
    with (store.directory / "writer.lock").open("a+") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if store.current() or store.gate.exists():
            raise RuntimeError("S6 owns this deployment; use sync_playbook instead of legacy setup")
        yield
