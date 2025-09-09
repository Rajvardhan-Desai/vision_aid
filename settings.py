from __future__ import annotations
import os
from typing import Optional, Callable, TypeVar
from dotenv import load_dotenv

# Load .env exactly once, as soon as the package is imported
load_dotenv()

_T = TypeVar("_T")

def _coerce(val: Optional[str], caster: Callable[[str], _T], default: _T) -> _T:
    if val is None or val == "":
        return default
    try:
        return caster(val)
    except Exception:
        return default

def env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def env_int(key: str, default: int = 0) -> int:
    return _coerce(os.getenv(key), int, default)

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None: return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"): return True
    if v in ("0", "false", "no", "n", "off"): return False
    return default

# Convenience email-related getters (typed)
def smtp_server() -> str:
    return env_str("SMTP_SERVER", "smtp.gmail.com")

def smtp_port() -> int:
    return env_int("SMTP_PORT", 587)

def sender_email() -> str:
    return env_str("SENDER_EMAIL", "")

def sender_password() -> str:
    return env_str("SENDER_PASSWORD", "")

def emergency_contact() -> str:
    return env_str("EMERGENCY_CONTACT", "")

def currency_model() -> str:
    return env_str("CURRENCY_MODEL", "best.onnx")
