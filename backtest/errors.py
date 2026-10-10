"""Errors a backtest request can fail with.

The desktop tester raised SystemExit for a bad strategy name or an unknown
parameter, which is right for a CLI and wrong inside a request handler:
SystemExit derives from BaseException, so Flask does not catch it and the
worker goes down over a typo in a query string. Everything that was a
SystemExit there is one of these here, and the routes turn them into a 400.
"""


class BacktestError(Exception):
    """A request the backtester cannot honour. The message is shown to the user."""


class InsufficientData(BacktestError):
    """Not enough usable bars to backtest."""
