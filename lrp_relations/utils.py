import hashlib
import socket
from pathlib import Path
from typing import Optional

import savethat


def find_credential_file() -> Optional[Path]:
    pkg_dir = Path(__file__).parent
    creds = pkg_dir.parent.parent / "savethat_credentials.toml"
    return creds if creds.exists() else None


def get_storage() -> savethat.Storage:
    return savethat.get_storage(
        "lrp_relations", credential_file=find_credential_file()
    )


def clevr_path() -> Path:
    host = socket.gethostname()
    if host == "goober":
        return Path("/srv/data/leonsixt/lrp_relations/data/CLEVR_v1.0")
    else:
        raise ValueError()


def clevr_xai_path() -> Path:
    host = socket.gethostname()
    if host == "goober":
        return Path("/srv/data/leonsixt/lrp_relations/data/CLEVR-XAI_v1.0")
    else:
        raise ValueError()


def sha256sum(filename: Path) -> str:
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()
