import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from prophet import Prophet
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

import data_source

# Silence Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def get_ml_insights(symbol):
    try:
        df = data_source.get_history(symbol, period="2y")
        if df.empty or len(df) < 50:
            return {"error": "Not enough historical data for ML models."}

        # 1. Calculate Pivots (Support, Resistance, Buy, Sell)
        last_day = df.iloc[-1]
        last_high = float(last_day['High'])
        last_low = float(last_day['Low'])
        last_close = float(last_day['Close'])
        
        pp = (last_high + last_low + last_close) / 3
        r1 = (pp * 2) - last_low
        s1 = (pp * 2) - last_high
        r2 = pp + (last_high - last_low)
        s2 = pp - (last_high - last_low)

        pivots = {
            "pivot": pp,
            "sell": r1,      # Target 1 / Sell Zone
            "buy": s1,       # Target 1 / Buy Zone
            "take_profit": r2, # Take Profit
            "stop_loss": s2    # Stop Loss
        }

        # 2. XGBoost for Next-Day Direction Probability
        df_full = df.copy()
        df_full['Returns'] = df_full['Close'].pct_change()
        df_full['SMA_10'] = df_full['Close'].rolling(window=10).mean()
        df_full['SMA_50'] = df_full['Close'].rolling(window=50).mean()
        
        # Target: 1 if next day close > today close, else 0
        df_full['Target'] = (df_full['Close'].shift(-1) > df_full['Close']).astype(int)
        
        features = ['Close', 'Returns', 'SMA_10', 'SMA_50', 'Volume']
        df_ml = df_full.copy()
        df_ml.dropna(inplace=True)
        
        if len(df_ml) > 100:
            X = df_ml[features]
            y = df_ml['Target']
            
            model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            model.fit(X, y)
            
            latest_row = df_full.iloc[-1:][features]
            prob_up = model.predict_proba(latest_row)[0][1]
        else:
            prob_up = 0.5

        # 3. Prophet Forecast (7-Days)
        df_prophet = df.copy()
        df_prophet.reset_index(inplace=True)
        # Ensure timezone-naive dates for Prophet
        if df_prophet['Date'].dt.tz is not None:
            df_prophet['ds'] = df_prophet['Date'].dt.tz_localize(None)
        else:
            df_prophet['ds'] = df_prophet['Date']
            
        df_prophet['y'] = df_prophet['Close']
        
        prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=True)
        prophet_model.fit(df_prophet[['ds', 'y']])
        future = prophet_model.make_future_dataframe(periods=7)
        forecast = prophet_model.predict(future)
        forecast_7d = forecast.iloc[-1]['yhat']

        # 4. Sentiment Analysis (Google News RSS + Vader)
        analyzer = SentimentIntensityAnalyzer()
        rss_url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        scores = []
        for entry in feed.entries[:20]:
            score = analyzer.polarity_scores(entry.title)['compound']
            scores.append(score)
            
        avg_sentiment = np.mean(scores) if scores else 0
        # Map from [-1, 1] to [0, 100] Fear & Greed index
        fear_greed_index = int((avg_sentiment + 1) * 50)

        return {
            "pivots": pivots,
            "xgboost_probability": float(prob_up),
            "prophet_7d_forecast": float(forecast_7d),
            "current_price": last_close,
            "fear_greed_index": fear_greed_index
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def get_advanced_signals(symbol):
    try:
        df = data_source.get_history(symbol, period="1y")
        if df.empty or len(df) < 20:
            return {"error": "Not enough data"}

        last_close = float(df['Close'].iloc[-1])
        
        # Calculate ATR (Average True Range) for dynamic Stop Loss / Take Profit
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = df['TR'].rolling(window=14).mean().iloc[-1]
        
        # Calculate Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        std_20 = df['Close'].rolling(window=20).std().iloc[-1]
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)

        # Basic Mean-Reversion / Trend-Following Signal
        # If price is near lower BB, it's a Buy signal. If near upper BB, Sell.
        signal = "HOLD"
        action = "none"
        if last_close <= lower_bb * 1.01:
            signal = "BUY"
            action = "buy"
        elif last_close >= upper_bb * 0.99:
            signal = "SELL"
            action = "sell"
        elif last_close > sma_20:
            signal = "WEAK BUY"
            action = "buy"
        else:
            signal = "WEAK SELL"
            action = "sell"

        # Calculate TP and SL based on ATR (e.g., risk 1 ATR, reward 2 ATR)
        if "BUY" in signal:
            sl = last_close - (1.5 * atr)
            tp = last_close + (3 * atr)
        else:
            sl = last_close + (1.5 * atr)
            tp = last_close - (3 * atr)

        return {
            "symbol": symbol,
            "current_price": last_close,
            "signal": signal,
            "action": action,
            "stop_loss": sl,
            "take_profit": tp,
            "atr": atr,
            "sma_20": sma_20
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

import scipy.optimize as sco

def portfolio_optimization(symbols):
    try:
        if not symbols or len(symbols) < 2:
            return {"error": "Need at least 2 symbols for optimization"}
            
        data = pd.DataFrame({s: data_source.get_close_series(s, "1y") for s in symbols}).dropna(how="all")
        if data.empty:
            return {"error": "Could not fetch data for symbols"}
            
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252 # Annualized
        cov_matrix = returns.cov() * 252 # Annualized
        num_assets = len(symbols)
        risk_free_rate = 0.04 # Assume 4% risk free rate for CAPM

        # Max Sharpe Ratio Optimization (Markowitz)
        def calc_portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_std_dev
            
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_ret, p_std = calc_portfolio_perf(weights, mean_returns, cov_matrix, risk_free_rate)
            return -(p_ret - risk_free_rate) / p_std if p_std != 0 else 0

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]

        opt_results = sco.minimize(neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                                   method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_weights = opt_results.x
        opt_ret, opt_std = calc_portfolio_perf(optimal_weights, mean_returns, cov_matrix, risk_free_rate)

        # CAPM Expected Returns
        market_data = data_source.get_close_series("^GSPC", "1y")  # S&P 500
        market_returns = market_data.pct_change().dropna()
        market_mean_return = market_returns.mean() * 252
        
        capm_returns = {}
        for idx, sym in enumerate(symbols):
            try:
                # Calculate Beta (align dates before covariance)
                aligned = pd.concat([returns[sym], market_returns], axis=1).dropna()
                cov_market = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1] * 252
                market_var = aligned.iloc[:, 1].var() * 252
                beta = cov_market / market_var
                
                # Expected Return = Rf + Beta * (E(Rm) - Rf)
                expected_return = risk_free_rate + beta * (market_mean_return - risk_free_rate)
                
                capm_returns[sym] = {
                    "beta": float(beta),
                    "expected_return": float(expected_return),
                    "optimal_weight": float(optimal_weights[idx])
                }
            except Exception:
                pass
                
        return {
            "portfolio_expected_annual_return": float(opt_ret),
            "portfolio_annual_volatility": float(opt_std),
            "portfolio_sharpe_ratio": float((opt_ret - risk_free_rate) / opt_std),
            "asset_details": capm_returns
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
