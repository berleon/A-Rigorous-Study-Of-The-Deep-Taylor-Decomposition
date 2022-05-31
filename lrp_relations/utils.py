import hashlib
import socket
from pathlib import Path
from typing import Optional

import numpy as np
import savethat
import toml
import torch
from savethat import env as env_mod


def find_credential_file() -> Optional[Path]:
    pkg_dir = Path(__file__).parent
    creds = pkg_dir.parent.parent / "savethat_credentials.toml"
    return creds if creds.exists() else None


def set_project_dir() -> None:
    pkg_dir = Path(__file__).parent
    env_mod.set_project_dir(pkg_dir.parent)


def get_credentials() -> env_mod.B2Credentials:
    pkg_dir = Path(__file__).parent
    cred_path = pkg_dir.parent.parent / "savethat_credentials.toml"
    with open(cred_path) as f:
        creds = toml.load(f)
    return env_mod.B2Credentials(**creds["lrp_relations"])


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


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
