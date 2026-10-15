"""Runtime configuration, read once from the environment.

Secrets come from `.env` (Neon, Gemini, Zerodha) and `.env.clerk` (Clerk), both
gitignored. Nothing here is ever logged or returned by a route.
"""
import base64
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(_ROOT, ".env.clerk"))


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.environ.get(name, default).split(",") if v.strip())


def _issuer_from_publishable_key(pk: str) -> str:
    """Clerk publishable keys are `pk_<env>_<base64("<frontend-api-host>$")>`."""
    try:
        encoded = pk.split("_", 2)[2]
        host = base64.b64decode(encoded + "=" * (-len(encoded) % 4)).decode().rstrip("$")
    except Exception:
        return ""
    return f"https://{host}" if host else ""


@dataclass(frozen=True)
class Settings:
    # Clerk. The issuer is the instance's Frontend API URL; tokens are verified
    # against its JWKS, or against CLERK_JWT_KEY (PEM) without a network call.
    clerk_issuer: str = ""
    clerk_jwt_key: str = ""
    clerk_secret_key: str = field(default="", repr=False)
    # Origins allowed to have minted a session token (the `azp` claim).
    authorized_parties: tuple[str, ...] = ()
    # Clerk user ids allowed to see operator diagnostics.
    admin_user_ids: tuple[str, ...] = ()

    # Neon. The app connects as a least-privilege role that row-level security
    # applies to; migrations use the owner URL.
    database_url_app: str = field(default="", repr=False)
    database_url_owner: str = field(default="", repr=False)

    max_body_bytes: int = 1024 * 1024

    # Zerodha Kite Connect (read-only holdings sync). The secret signs the
    # request-token exchange; the token key (32 bytes, base64) encrypts stored
    # access tokens. Broker routes answer 503 until all three are set.
    zerodha_api_key: str = field(default="", repr=False)
    zerodha_api_secret: str = field(default="", repr=False)
    broker_token_key: str = field(default="", repr=False)

    @property
    def zerodha_configured(self) -> bool:
        return bool(self.zerodha_api_key and self.zerodha_api_secret and self.broker_token_key)

    @property
    def clerk_jwks_url(self) -> str:
        return f"{self.clerk_issuer}/.well-known/jwks.json" if self.clerk_issuer else ""


def load() -> Settings:
    pk = (os.environ.get("CLERK_PUBLISHABLE_KEY")
          or os.environ.get("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY") or "")
    return Settings(
        clerk_issuer=(os.environ.get("CLERK_ISSUER") or _issuer_from_publishable_key(pk)).rstrip("/"),
        clerk_jwt_key=os.environ.get("CLERK_JWT_KEY", "").replace("\\n", "\n"),
        clerk_secret_key=os.environ.get("CLERK_SECRET_KEY", ""),
        authorized_parties=_csv("AUTHORIZED_PARTIES", "http://localhost:3000,http://localhost:3123"),
        admin_user_ids=_csv("ADMIN_USER_IDS", ""),
        database_url_app=os.environ.get("DATABASE_URL_APP", ""),
        database_url_owner=os.environ.get("DATABASE_URL_UNPOOLED", ""),
        max_body_bytes=int(os.environ.get("MAX_CONTENT_LENGTH_BYTES", 1024 * 1024)),
        zerodha_api_key=os.environ.get("ZERODHA_API_KEY", ""),
        zerodha_api_secret=os.environ.get("ZERODHA_API_SECRET", ""),
        broker_token_key=os.environ.get("BROKER_TOKEN_KEY", ""),
    )


settings = load()
