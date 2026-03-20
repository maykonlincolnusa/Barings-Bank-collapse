from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.configs.settings import settings
from app.utils.paths import ENCRYPTED_ARTIFACT_PATH


def maybe_encrypt_artifact(path: Path) -> Path | None:
    if not settings.encryption_key:
        return None
    raw_key = _resolve_key(settings.encryption_key)
    aesgcm = AESGCM(raw_key)
    nonce = os.urandom(12)
    plaintext = path.read_bytes()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    ENCRYPTED_ARTIFACT_PATH.write_bytes(nonce + ciphertext)
    return ENCRYPTED_ARTIFACT_PATH


def _resolve_key(value: str) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(value.encode("utf-8"))
    except Exception:
        decoded = value.encode("utf-8")
    if len(decoded) >= 32:
        return decoded[:32]
    return (decoded * (32 // len(decoded) + 1))[:32]

