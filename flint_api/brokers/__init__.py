"""Read-only broker adapters, by name. Order is the order shown in Settings."""
from . import angel, binance, dhan, fyers, groww, indmoney, upstox, zerodha
from .base import Session, TokenExpired

ADAPTERS = {m.NAME: m for m in (zerodha, upstox, groww, angel, fyers, dhan, indmoney, binance)}

__all__ = ["ADAPTERS", "Session", "TokenExpired"]
