"""Input types validated at the edge, before any engine sees them.

A raw caller-chosen symbol used to become a cache key, a dict key and an
outbound URL across 13 routes (audit findings 5, 6, 9, 10). Constraining its
shape here bounds all of them at once.
"""
from typing import Annotated

from pydantic import AfterValidator, StringConstraints

# Yahoo-style symbols: AAPL, BRK-B, RELIANCE.NS, M&M.NS, ^GSPC, EURUSD=X, GC=F, BTC-USD
SYMBOL_PATTERN = r"^[A-Za-z0-9^][A-Za-z0-9.\-=&^]{0,23}$"

Symbol = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=SYMBOL_PATTERN),
    AfterValidator(str.upper),
]

# Registry ids (universes, screens, scopes, strategies) are closed sets checked
# by the engines; this only bounds their shape and length.
RegistryId = Annotated[str, StringConstraints(strip_whitespace=True,
                                              pattern=r"^[a-z0-9_\-]{1,48}$")]

MAX_SYMBOLS_PER_REQUEST = 12
