from flask import Flask, Response, request, jsonify, stream_with_context
import json
import os
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import requests
import ml_models
import geotrade_layer
import quant_models
import news_engine
import news_intel
import fundamentals
import deep_analysis
import ai_assistant
import risk_analytics
import data_source
import screener_engine
import levels_engine
import nse_segments
import movers_engine
import agents_desk
import llm_router
from backtest import service as backtest_service
from backtest.errors import BacktestError
from rate_limiter import rate_limit

app = Flask(__name__)

# Allow CORS for the Next.js frontend. Origins are an explicit allowlist from
# CORS_ORIGINS (comma-separated) so a deployment names its own frontend without
# a code change; the default covers local dev on either loopback spelling.
#
# This was previously "*". These routes carry no auth, so a wildcard let any
# page a user visited drive the whole API from their browser — spending Gemini
# quota and, because the limiter buckets by client IP, drawing on a different
# bucket per victim rather than the attacker's own.
_DEFAULT_CORS_ORIGINS = 'http://localhost:3000,http://127.0.0.1:3000'
CORS_ORIGINS = [o.strip() for o in
                os.environ.get('CORS_ORIGINS', _DEFAULT_CORS_ORIGINS).split(',')
                if o.strip()]

# expose_headers: Retry-After / X-RateLimit-* are not CORS-safelisted, so
# without this the browser hides them from fetch() and the client can't
# show an accurate rate-limit countdown.
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}},
     expose_headers=['Retry-After', 'X-RateLimit-Limit',
                     'X-RateLimit-Remaining', 'X-RateLimit-Reset'])

# Cap request bodies. The POST routes take JSON of bounded shape (a position
# list, a chat history, a backtest spec), so a megabyte is already generous;
# without a cap a single request can grow the worker's memory without limit.
app.config['MAX_CONTENT_LENGTH'] = int(
    os.environ.get('MAX_CONTENT_LENGTH_BYTES', 1 * 1024 * 1024))

# A 500 means an unexpected internal failure. The exception text routinely
# carries upstream URLs, local file paths and library internals, so it goes to
# the log and the client gets a fixed string. The deliberate 4xx/503 messages
# raised by the engines' own exception types are safe to show and keep theirs.
_INTERNAL_ERROR = 'Internal error — the request could not be completed.'


def _server_error(exc):
    """Log the real cause, return an opaque 500 body."""
    app.logger.exception('unhandled error on %s', request.path)
    return jsonify({'error': _INTERNAL_ERROR}), 500


@app.errorhandler(413)
def _too_large(_e):
    return jsonify({'error': 'Request body too large.'}), 413

# Client-side caching policy per resource class (system-design: cache close
# to the consumer; the browser absorbs repeat GETs before they hit us).
_CACHE_POLICY = {
    '/api/quote': 'public, max-age=30',
    '/api/chart': 'public, max-age=300',
    '/api/search': 'public, max-age=300',
    '/api/trending': 'public, max-age=120',
    '/api/market-overview': 'public, max-age=120',
    '/api/news': 'public, max-age=300',
    '/api/news-intel': 'public, max-age=300',
    '/api/fundamentals': 'public, max-age=3600',
    '/api/ml-insights': 'public, max-age=600',
    '/api/quant-signals': 'public, max-age=300',
    '/api/discover': 'public, max-age=300',
    '/api/screens': 'public, max-age=3600',
    '/api/screens/run': 'public, max-age=300',
    # Movers is polled while the price cache warms, so it caches only briefly —
    # and the route drops to no-store entirely while the board is partial.
    '/api/movers': 'public, max-age=60',
    '/api/movers/scopes': 'public, max-age=3600',
    '/api/backtest/strategies': 'public, max-age=3600',
}

@app.after_request
def add_cache_headers(response):
    if request.method == 'GET' and response.status_code == 200:
        policy = _CACHE_POLICY.get(request.path)
        # Never let the browser cache a failure payload: some handlers return
        # {"error": ...} with HTTP 200 (stale-if-error semantics), and caching
        # those would keep showing the outage after the backend recovers.
        if policy and not (response.is_json and b'"error"' in response.get_data()[:120]):
            response.headers.setdefault('Cache-Control', policy)
    return response


@app.after_request
def add_security_headers(response):
    """Baseline hardening headers.

    This process answers with JSON and SSE only — it serves no HTML, script or
    frame — so the CSP can deny everything outright rather than enumerate
    sources. nosniff matters most here: it stops a browser treating a JSON
    body that happens to contain markup as HTML.

    No HSTS: this binds plain http:// locally, where browsers ignore the header
    anyway. If this ever sits behind a TLS-terminating proxy, that proxy is the
    only place that knows the real scheme and should own the policy for the
    whole domain.
    """
    response.headers.setdefault('Content-Security-Policy',
                                "default-src 'none'; frame-ancestors 'none'")
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    return response

@app.route('/api/search', methods=['GET'])
@rate_limit('standard')
def search():
    query = request.args.get('q')
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    try:
        response = data_source.http_get(url, {'q': query, 'quotesCount': 5})
        data = response.json()
        return jsonify(data)
    except Exception as e:
        return _server_error(e)

@app.route('/api/quote', methods=['GET'])
@rate_limit('standard')
def quote():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400
    
    try:
        return jsonify(data_source.get_quote(symbol))
    except Exception as e:
        return _server_error(e)

@app.route('/api/chart', methods=['GET'])
@rate_limit('standard')
def chart():
    """OHLCV candles at any supported interval.

    `range` may exceed what Yahoo allows for the interval; it is clamped to the
    legal maximum (and the clamped value is reported back) instead of 422-ing.

    `warmup` is the point of the range/interval split: a 200-period EMA over a
    1y view needs ~200 bars BEFORE that view or the line only starts 200 bars in,
    which is what made long indicators look like they stopped early. The client
    asks for extra history, then shows only the tail. Indicators are computed on
    everything returned, so the overlay is continuous across every visible bar
    without fabricating warm-up values.
    """
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400

    interval = request.args.get('interval', '1d')
    if interval not in data_source.VALID_INTERVALS:
        return jsonify({
            'error': f'Unsupported interval "{interval}". '
                     f'Valid: {", ".join(data_source.VALID_INTERVALS)}'
        }), 400

    requested_range = request.args.get('range', '1y')
    effective_range = data_source.clamp_range(interval, requested_range)

    try:
        records = data_source.get_ohlc(symbol, range_=effective_range, interval=interval)

        # Intraday bars need a full timestamp; daily and coarser collapse to a
        # date so lightweight-charts treats them as business days.
        intraday = interval not in ('1d', '1wk', '1mo')
        quotes = []
        for r in records:
            ts = pd.Timestamp(r['t'])
            quotes.append({
                'time': int(ts.timestamp()) if intraday else ts.strftime('%Y-%m-%d'),
                'open': r['o'], 'high': r['h'], 'low': r['l'], 'close': r['c'],
                'volume': r['v'],
            })

        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'range': effective_range,
            # Surfaced so the UI can say "showing 2y, you asked for 5y" rather
            # than silently serving less than requested.
            'range_clamped': effective_range != requested_range,
            'requested_range': requested_range,
            'intraday': intraday,
            'count': len(quotes),
            'quotes': quotes,
        })
    except Exception as e:
        return _server_error(e)

@app.route('/api/trending', methods=['GET'])
@rate_limit('standard')
def trending():
    try:
        return jsonify(data_source.get_trending())
    except Exception as e:
        return _server_error(e)

@app.route('/api/ml-insights', methods=['GET'])
@rate_limit('standard')
def ml_insights():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400
    
    insights = ml_models.get_ml_insights(symbol)
    return jsonify(insights)

@app.route('/api/advanced-signals', methods=['GET'])
@rate_limit('standard')
def advanced_signals():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400
    
    signals = ml_models.get_advanced_signals(symbol)
    return jsonify(signals)

@app.route('/api/quant-signals', methods=['GET'])
@rate_limit('standard')
def quant_signals():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400

    return jsonify(quant_models.get_quant_signals(symbol))

@app.route('/api/news', methods=['GET'])
@rate_limit('standard')
def global_news():
    return jsonify(news_engine.get_global_news())

@app.route('/api/news/ticker', methods=['GET'])
@rate_limit('standard')
def ticker_news():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400

    return jsonify(news_engine.get_ticker_news(symbol))

@app.route('/api/news-intel', methods=['GET'])
@rate_limit('standard')
def ticker_news_intel():
    """Multi-source news sweep + finance-tuned sentiment rating for one ticker."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400

    return jsonify(news_intel.get_news_intel(symbol))

@app.route('/api/fundamentals', methods=['GET'])
@rate_limit('standard')
def fundamentals_xray():
    """Annual statements, derived ratios, red flags, valuation vs own history."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400

    return jsonify(fundamentals.get_fundamentals(symbol))

@app.route('/api/deep-analysis', methods=['GET'])
@rate_limit('expensive')
def deep_analysis_route():
    """Full teardown: fundamentals + technical phase + news + composite verdict.
    Pass ai=1 to also generate the Gemini research-note narrative (slower)."""
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'Query parameter "symbol" is required'}), 400

    include_ai = request.args.get('ai') == '1'
    return jsonify(deep_analysis.get_deep_analysis(symbol, include_ai=include_ai))

@app.route('/api/holdings-intel', methods=['POST'])
@rate_limit('expensive')
def holdings_intel():
    """Batch compact intel (health, stance, phase, verdict) for held positions."""
    data = request.json or {}
    positions = data.get('positions', [])
    if not positions:
        return jsonify({'error': 'JSON body must contain a "positions" list'}), 400

    return jsonify(deep_analysis.get_holdings_intel(positions))

@app.route('/api/ai-chat', methods=['POST'])
@rate_limit('expensive')
def ai_chat():
    data = request.json or {}
    messages = data.get('messages', [])
    if not messages:
        return jsonify({'error': 'JSON body must contain a non-empty "messages" list'}), 400

    result = ai_assistant.chat(
        messages=messages,
        portfolio=data.get('portfolio'),
        symbol=data.get('symbol')
    )
    return jsonify(result)

@app.route('/api/portfolio-analytics', methods=['POST'])
@rate_limit('standard')
def portfolio_analytics():
    data = request.json or {}
    positions = data.get('positions', [])
    balance = data.get('balance', 0)
    if not positions:
        return jsonify({'error': 'JSON body must contain a "positions" list'}), 400

    return jsonify(risk_analytics.get_portfolio_analytics(positions, balance))

@app.route('/api/portfolio-optimization', methods=['POST'])
@rate_limit('expensive')
def portfolio_optimization():
    data = request.json
    symbols = data.get('symbols', [])
    if not symbols or not isinstance(symbols, list):
        return jsonify({'error': 'JSON body must contain a "symbols" list'}), 400
        
    result = ml_models.portfolio_optimization(symbols)
    return jsonify(result)

@app.route('/api/market-overview', methods=['GET'])
@rate_limit('standard')
def market_overview():
    try:
        return jsonify(data_source.get_market_overview())
    except Exception as e:
        return _server_error(e)

def _scan_response(result):
    """jsonify a scan payload; an outage scan (parsed == 0) must not be
    browser-cached — the engine already refuses to memoise it, and a
    Cache-Control'd 200 would pin the empty page past upstream recovery.
    setdefault in add_cache_headers lets this explicit header win."""
    resp = jsonify(result)
    if result.get('parsed') == 0:
        resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/api/discover', methods=['GET'])
@rate_limit('standard')
def discover():
    """Horizon-tagged idea cards for a universe (listed alphabetically — never ranked)."""
    universe = request.args.get('universe', 'us_large')
    # all=1 (the Markets hub): a card for every parsed symbol, tags may be [].
    include_untagged = request.args.get('all') == '1'
    try:
        return _scan_response(
            screener_engine.get_discover(universe, include_untagged=include_untagged))
    except screener_engine.UnknownUniverseError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return _server_error(e)

@app.route('/api/screens', methods=['GET'])
@rate_limit('standard')
def screens_catalogue():
    """Prebuilt screen catalogue — static registry, conditions text derived
    from the same specs the evaluator runs."""
    try:
        return jsonify(screener_engine.list_screens())
    except Exception as e:
        return _server_error(e)

@app.route('/api/screens/run', methods=['GET'])
@rate_limit('standard')
def screens_run():
    """Run one prebuilt screen over a universe."""
    screen_id = request.args.get('id')
    if not screen_id:
        return jsonify({'error': 'Query parameter "id" is required'}), 400

    universe = request.args.get('universe', 'us_large')
    try:
        return _scan_response(screener_engine.run_screen(screen_id, universe))
    except screener_engine.UnknownScreenError as e:
        return jsonify({'error': str(e)}), 404
    except screener_engine.UnknownUniverseError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return _server_error(e)

@app.route('/api/levels', methods=['GET'])
@rate_limit('standard')
def levels():
    """Names sitting at a respected daily/weekly support or resistance level.

    Runs off the same cached universe series Discover scans, so hitting this
    for a universe already loaded costs no upstream calls. Ranked by level
    quality and proximity — the one Discovery surface that deliberately does
    rank, because "which names are at the best levels" is the question.
    """
    universe = request.args.get('universe', 'us_large')
    side = request.args.get('side', 'all')
    try:
        limit = int(request.args.get('limit', 40))
    except ValueError:
        return jsonify({'error': 'Query parameter "limit" must be an integer'}), 400

    try:
        return _scan_response(levels_engine.get_levels(universe, limit=limit, side=side))
    except screener_engine.UnknownUniverseError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return _server_error(e)


@app.route('/api/movers/scopes', methods=['GET'])
@rate_limit('standard')
def movers_scopes():
    """Segment/index picker for the India movers board. Membership only, no
    pricing, so this stays fast even while the price cache is cold."""
    try:
        return jsonify(movers_engine.get_scopes())
    except nse_segments.SegmentsUnavailableError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return _server_error(e)


@app.route('/api/movers', methods=['GET'])
@rate_limit('standard')
def movers():
    """Cap-segment movers, circuit hits and intraday leaders for one NSE scope.

    A cold cache returns a PARTIAL board immediately (with coverage reported)
    and warms the rest in the background, so this never blocks on a ~3 min scan.
    Partial payloads must not be browser-cached or the incomplete board would
    outlive the warm.

    Defaults to 'allnse' rather than an index: a 20% circuit is almost always
    outside the index lists, so defaulting to a narrower scope would hide the
    very rows this board exists to show.
    """
    scope = request.args.get('scope', 'allnse')
    try:
        min_move = float(request.args.get('min_move', movers_engine.DEFAULT_MIN_MOVE))
    except ValueError:
        return jsonify({'error': 'Query parameter "min_move" must be a number'}), 400
    try:
        limit = int(request.args.get('limit', 20))
    except ValueError:
        return jsonify({'error': 'Query parameter "limit" must be an integer'}), 400

    try:
        result = movers_engine.get_movers(scope, min_move_pct=min_move, limit=limit)
    except nse_segments.UnknownScopeError as e:
        return jsonify({'error': str(e)}), 400
    except nse_segments.SegmentsUnavailableError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return _server_error(e)

    resp = jsonify(result)
    if result['partial'] or result['priced'] == 0:
        resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/api/analyze-idea', methods=['POST'])
@rate_limit('standard')
def analyze_idea():
    data = request.json
    text = data.get('title', '') + " " + data.get('content', '')
    ticker = data.get('ticker', '')
    sentiment = data.get('sentiment', 'BULLISH')
    
    # 1. Base Sentiment Analysis of user's justification
    sentiment_score = ml_models.sentiment_analyzer.polarity_scores(text)['compound']
    
    # 2. Risk Context based on Text Length & Detail
    word_count = len(text.split())
    
    # Simple algorithmic confidence heuristic substituting a full neural-net here
    # Higher word count + stronger sentiment == higher confidence
    base_confidence = 50
    confidence_modifier = (sentiment_score * 20) if sentiment == 'BULLISH' else (sentiment_score * -20)
    word_count_modifier = min(20, word_count / 10) 
    
    final_confidence = min(99, max(1, int(base_confidence + confidence_modifier + word_count_modifier)))
    
    risk_analysis = "Moderate Risk"
    if final_confidence < 30:
        risk_analysis = "High Risk: Lacking supporting rationale or conflicting text sentiment."
    elif final_confidence > 75:
        risk_analysis = "Low Risk: Strong supporting rationale aligned with directional bias."

    return jsonify({
        'confidence_score': final_confidence,
        'risk_analysis': risk_analysis
    })

@app.route('/api/geotrade', methods=['GET'])
@rate_limit('expensive')
def geotrade():
    try:
        data = geotrade_layer.get_geotrade_analysis()
        return jsonify(data)
    except Exception as e:
        return _server_error(e)

@app.route('/api/geotrade/deep', methods=['GET'])
@rate_limit('expensive')
def geotrade_deep():
    country = request.args.get('country')
    if not country:
        return jsonify({'error': 'Country code is required'}), 400
        
    try:
        data = geotrade_layer.get_deep_country_analysis(country)
        return jsonify(data)
    except Exception as e:
        return _server_error(e)

# ------------------------------------------------------------ trading desk ---
# Multi-agent decision pipeline (see agents_desk.py): four analysts, a bull/bear
# debate, a trader, a three-seat risk committee and a portfolio manager, with a
# journal of past decisions scored on realized alpha feeding the next run.

@app.route('/api/agents/run', methods=['POST'])
@rate_limit('agents')
def agents_run():
    """Stream a full desk run as Server-Sent Events.

    POST rather than GET (so the paper portfolio travels in the body, not a
    query string), which is why the client reads the stream with fetch() rather
    than EventSource. Pass {"stream": false} to get one collected JSON payload
    instead — the run still takes the same ~40s, it just arrives at the end.
    """
    data = request.json or {}
    symbol = (data.get('symbol') or '').strip()
    if not symbol:
        return jsonify({'error': 'JSON body must contain a "symbol"'}), 400

    try:
        rounds = int(data.get('rounds', agents_desk.DEFAULT_ROUNDS))
    except (TypeError, ValueError):
        return jsonify({'error': 'Body field "rounds" must be an integer'}), 400
    if not 1 <= rounds <= agents_desk.MAX_ROUNDS:
        return jsonify({
            'error': f'"rounds" must be between 1 and {agents_desk.MAX_ROUNDS}'}), 400

    portfolio = data.get('portfolio') if isinstance(data.get('portfolio'), dict) else None

    if data.get('stream') is False:
        return jsonify(agents_desk.run_desk_sync(symbol, portfolio=portfolio, rounds=rounds))

    def generate():
        # Every event is emitted inside run_desk's own try/except, so a failure
        # arrives as an {"type": "error"} frame rather than a truncated stream.
        for event in agents_desk.run_desk(symbol, portfolio=portfolio, rounds=rounds):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    resp = Response(stream_with_context(generate()), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['Connection'] = 'keep-alive'
    # Proxies that buffer would defeat the point of streaming the debate.
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/api/agents/routing', methods=['GET'])
@rate_limit('standard')
def agents_routing():
    """Where desk calls are currently going: Gemini, or the OmniRoute fallback.

    Reports whether the gateway is reachable and how long Gemini is benched for,
    so an operator can tell "quota exhausted, rerouted" apart from "quota
    exhausted, no fallback installed".
    """
    return jsonify(llm_router.status())


@app.route('/api/agents/journal', methods=['GET'])
@rate_limit('standard')
def agents_journal():
    """Past desk decisions, scored against realized return and alpha vs the S&P.

    Scoring runs off data_source's cached closes, so this is cheap to refresh.
    """
    symbol = request.args.get('symbol')
    try:
        limit = int(request.args.get('limit', 60))
    except ValueError:
        return jsonify({'error': 'Query parameter "limit" must be an integer'}), 400

    try:
        return jsonify(agents_desk.get_journal(symbol, limit=max(1, min(limit, 200))))
    except Exception as e:
        return _server_error(e)


@app.route('/api/agents/journal/<entry_id>', methods=['DELETE'])
@rate_limit('standard')
def agents_journal_delete(entry_id):
    """Drop one decision from the journal so it stops informing future runs."""
    if agents_desk.journal_delete(entry_id):
        return jsonify({'deleted': entry_id})
    return jsonify({'error': 'No journal entry with that id.'}), 404


# ------------------------------------------------------------- backtesting ---
# Event-driven backtester (see backtest/): signals formed at a bar's close and
# filled at the next open, costs charged on both sides, stops checked intrabar
# against the real high and low. The engines are the desktop tester's, so a run
# here and a run there agree bar for bar on the same frame.

@app.route('/api/backtest/strategies', methods=['GET'])
@rate_limit('standard')
def backtest_strategies():
    """The strategy catalogue: names, one-line descriptions and every parameter."""
    return jsonify({'strategies': backtest_service.list_strategies()})


@app.route('/api/backtest/run', methods=['POST'])
@rate_limit('backtest')
def backtest_run():
    """One backtest, plus whichever robustness passes were requested.

    POST rather than GET: the body carries a strategy parameter map, a date
    window and the cost model, which do not belong in a query string.

    Synchronous — a daily backtest over four decades of bars is a couple of
    seconds once the history is cached. The sweeps below are the ones that
    stream, because they are hundreds of these.
    """
    req = request.json or {}
    try:
        return jsonify(backtest_service.run(req))
    except BacktestError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return _server_error(e)


def _stream_backtest(generator_fn, req):
    """Run one of the sweep generators as Server-Sent Events.

    The first event is produced inside this handler rather than in the stream,
    so a bad grid or an unknown symbol still comes back as a 400 with a real
    message instead of a 200 whose first frame happens to be an error.
    """
    try:
        stream = generator_fn(req)
        first = next(stream)
    except BacktestError as e:
        return jsonify({'error': str(e)}), 400
    except StopIteration:
        return jsonify({'error': 'The sweep produced no work to do.'}), 400
    except Exception as e:
        return _server_error(e)

    def generate():
        yield f"data: {json.dumps(first, default=str)}\n\n"
        try:
            for event in stream:
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except BacktestError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception:
            app.logger.exception('unhandled error streaming %s', request.path)
            yield f"data: {json.dumps({'type': 'error', 'message': _INTERNAL_ERROR})}\n\n"

    resp = Response(stream_with_context(generate()), mimetype='text/event-stream')
    resp.headers['Cache-Control'] = 'no-store'
    resp.headers['Connection'] = 'keep-alive'
    # Proxies that buffer would defeat the point of streaming the progress.
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.route('/api/backtest/optimize', methods=['POST'])
@rate_limit('backtest')
def backtest_optimize():
    """Parameter sweep, streamed. Ranked by Sharpe, with a note on whether the
    winner sits on a plateau or is a lone spike in the noise."""
    return _stream_backtest(backtest_service.optimize, request.json or {})


@app.route('/api/backtest/walk-forward', methods=['POST'])
@rate_limit('backtest')
def backtest_walk_forward():
    """Rolling optimise-then-trade folds, streamed one fold at a time.

    The expensive one: the whole grid runs once per fold. It is also the only
    result here that was never fitted on the bars it reports.
    """
    return _stream_backtest(backtest_service.walk_forward, request.json or {})


if __name__ == '__main__':
    # debug must be opted into explicitly: Flask's debug mode serves the
    # Werkzeug console, which executes arbitrary Python from anyone who can
    # reach it. run.sh launches this path, so defaulting it on exposed the
    # console on every local run. Set FLASK_DEBUG=1 when you want it.
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', port=5000)
