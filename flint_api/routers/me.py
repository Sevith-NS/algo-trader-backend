"""The signed-in user: profile, onboarding, data export, account deletion."""
import re
from decimal import Decimal
from typing import Annotated, Literal

import requests
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import db
from ..errors import AppError, Conflict, Forbidden, Unavailable
from ..ratelimit import limit
from ..security import Principal
from ..settings import settings

router = APIRouter(prefix="/me", tags=["me"])

Currency = Literal["INR", "USD", "EUR", "GBP", "JPY", "SGD", "AUD", "CAD", "CHF", "HKD", "AED"]
Market = Literal["india", "us", "europe", "asia", "crypto", "fx", "commodities", "bonds"]
RiskPct = Annotated[Decimal, Field(gt=0, le=10, max_digits=4, decimal_places=2)]

_CLERK_API = "https://api.clerk.com/v1"
_CLERK_USER_ID = re.compile(r"^user_[A-Za-z0-9]{1,64}$")


def _profile(conn) -> dict:
    row = conn.execute("select id, plan, base_currency, risk_per_trade_pct, markets, onboarded_at, created_at "
                       "from users").fetchone()
    return {**row, "risk_per_trade_pct": str(row["risk_per_trade_pct"]),
            "onboarded": row["onboarded_at"] is not None}


@router.get("")
def me(user: Principal = Depends(limit("standard"))):
    with db.user_tx(user.user_id) as conn:
        return _profile(conn)


class PreferencesBody(BaseModel):
    base_currency: Currency | None = None
    risk_per_trade_pct: RiskPct | None = None
    markets: Annotated[list[Market], Field(max_length=8)] | None = None


@router.patch("")
def update_preferences(body: PreferencesBody, user: Principal = Depends(limit("write"))):
    changes = body.model_dump(exclude_none=True)
    if "markets" in changes:
        changes["markets"] = sorted(set(changes["markets"]))
    with db.user_tx(user.user_id) as conn:
        for column, value in changes.items():
            if column not in _EDITABLE:  # names come from the model; this keeps it that way
                raise AppError("Unknown preference.")
            conn.execute(f"update users set {column} = %s", (value,))
        return _profile(conn)


_EDITABLE = frozenset({"base_currency", "risk_per_trade_pct", "markets"})


class OnboardingBody(BaseModel):
    base_currency: Currency
    markets: Annotated[list[Market], Field(min_length=1, max_length=8)]
    risk_per_trade_pct: RiskPct = Decimal("1.00")
    starting_cash: Annotated[Decimal, Field(gt=0, le=Decimal("1e11"), max_digits=15, decimal_places=2)]


@router.post("/onboarding")
def onboarding(body: OnboardingBody, user: Principal = Depends(limit("write"))):
    with db.user_tx(user.user_id) as conn:
        if conn.execute("select onboarded_at from users").fetchone()["onboarded_at"] is not None:
            raise Conflict("You've already finished setup. Change preferences in Settings.")
        conn.execute("update users set base_currency = %s, markets = %s, risk_per_trade_pct = %s, "
                     "onboarded_at = now()",
                     (body.base_currency, sorted(set(body.markets)), body.risk_per_trade_pct))
        conn.execute("insert into accounts (user_id, name, currency, starting_cash) values (%s, 'Paper', %s, %s)",
                     (user.user_id, body.base_currency, body.starting_cash))
        return _profile(conn)


@router.get("/export")
def export(user: Principal = Depends(limit("expensive"))):
    """Everything Flint stores about the caller, as one JSON download."""
    with db.user_tx(user.user_id) as conn:
        data = {
            "profile": _profile(conn),
            "accounts": conn.execute("select * from accounts order by created_at").fetchall(),
            "fills": conn.execute("select * from fills order by filled_at").fetchall(),
            "watchlists": conn.execute("select * from watchlists order by created_at").fetchall(),
            "watchlist_items": conn.execute("select * from watchlist_items order by added_at").fetchall(),
            "journal_entries": conn.execute("select * from journal_entries order by created_at").fetchall(),
            # Nothing from brokers: connections live only in the user's browser.
            "alerts": conn.execute("select * from alerts order by created_at").fetchall(),
            "alert_firings": conn.execute("select * from alert_firings order by fired_at").fetchall(),
        }
    import json
    body = json.loads(json.dumps(data, default=str))
    return JSONResponse(body, headers={"Content-Disposition": 'attachment; filename="flint-export.json"'})


class DeleteBody(BaseModel):
    confirm_email: Annotated[str, Field(min_length=3, max_length=320)]


def _clerk(method: str, user_id: str) -> requests.Response:
    if not settings.clerk_secret_key or not _CLERK_USER_ID.match(user_id):
        raise Unavailable("Account deletion isn't available right now.")
    try:
        return requests.request(method, f"{_CLERK_API}/users/{user_id}", timeout=10,
                                headers={"Authorization": f"Bearer {settings.clerk_secret_key}"})
    except requests.RequestException:
        raise Unavailable("Account deletion isn't available right now. Try again shortly.")


@router.delete("")
def delete_account(body: DeleteBody, user: Principal = Depends(limit("write"))):
    """Delete all Flint data and the Clerk account. The caller must type their
    email address, matched against Clerk's record rather than anything sent."""
    profile = _clerk("GET", user.user_id)
    if profile.status_code != 200:
        raise Unavailable("Account deletion isn't available right now. Try again shortly.")
    emails = {e.get("email_address", "").strip().lower()
              for e in profile.json().get("email_addresses", [])}
    if body.confirm_email.strip().lower() not in emails:
        raise Forbidden("That email doesn't match your account.")

    # Data first, inside a transaction that only commits if Clerk also deletes
    # the identity, so a failure never leaves an identity without its data
    # policy applied or data with no owner.
    with db.user_tx(user.user_id) as conn:
        conn.execute("delete from users where id = %s", (user.user_id,))
        gone = _clerk("DELETE", user.user_id)
        if gone.status_code not in (200, 404):
            raise AppError("We couldn't delete your sign-in account, so nothing was deleted. Try again.")
    db.forget_user(user.user_id)
    return {"deleted": True}
