import os
import time
import json
import feedparser
import traceback
import urllib.parse
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables (look in parent directories if needed)
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Fallback API key loaded from .env if needed
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Using ISO-3 Codes for react-globe.gl compatibility
TARGET_COUNTRIES = {
    "USA": "United States",
    "CHN": "China",
    "IND": "India",
    "GBR": "United Kingdom",
    "DEU": "Germany",
    "JPN": "Japan",
    "FRA": "France",
    "BRA": "Brazil",
    "CAN": "Canada",
    "AUS": "Australia",
    "RUS": "Russia",
    "ZAF": "South Africa"
}

def fetch_country_headlines(country_name, max_articles=4):
    """Fetches the latest economic headlines for a given country."""
    encoded_country = urllib.parse.quote(country_name)
    url = f"https://news.google.com/rss/search?q={encoded_country}+economy+OR+market&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    headlines = []
    for entry in feed.entries[:max_articles]:
        headlines.append(entry.title)
    return headlines

def get_geotrade_analysis():
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not found in environment variables. Please add it to your .env file."}
        
    try:
        grouped_headlines = {}
        for country_code, country_name in TARGET_COUNTRIES.items():
            headlines = fetch_country_headlines(country_name)
            grouped_headlines[country_code] = headlines
            
        prompt = f"""
        You are a highly advanced systematic global macro trading AI. 
        Analyze the following recent news headlines for various countries. 
        For each country, provide a sentiment score (-1.0 to 1.0) regarding their immediate economic/market outlook.
        Also provide a short 2-3 word sentiment label (e.g., 'Strongly Bullish', 'Neutral', 'Slightly Bearish') and an array of 2 very brief, actionable trade ideas (e.g., "Long XYZ", "Short ABC").
        
        Headlines:
        {json.dumps(grouped_headlines, indent=2)}
        
        Respond ONLY with a valid stringified JSON array format (do not use markdown blocks like ```json). It must be parseable.
        Example item format: 
        [
            {{
                "countryCode": "USA",
                "score": 0.85,
                "label": "Strongly Bullish",
                "trades": ["Long S&P", "Short VIX"]
            }}
        ]
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        data = json.loads(response_text)
        return data
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

def get_deep_country_analysis(country_code):
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not found in environment variables. Please add it to your .env file."}
        
    country_name = TARGET_COUNTRIES.get(country_code)
    if not country_name:
        return {"error": f"Country code {country_code} is not supported."}
        
    try:
        # Fetch up to 10 articles for a deep dive
        headlines = fetch_country_headlines(country_name, max_articles=8)
        
        prompt = f"""
        You are an expert global macro quantitative analyst.
        I am providing you with the latest major news headlines regarding the economy and markets of {country_name}.
        
        Headlines:
        {json.dumps(headlines, indent=2)}
        
        Using only this data and your innate macroeconomic knowledge, output a highly detailed analysis in the following strict JSON format (do not use markdown blocks like ```json):
        {{
            "countryName": "{country_name}",
            "thesis": "A 3-4 paragraph long-form macroeconomic thesis covering monetary policy outlook, market sentiment, and key risks.",
            "headlines": {json.dumps(headlines)},
            "recommendations": [
                {{
                    "symbol": "TICKER",
                    "name": "ETF or Stock Name",
                    "rationale": "1 sentence explanation of why this asset is recommended based on the thesis (e.g. going Long or Short)."
                }}
            ]
        }}
        
        Ensure "recommendations" contains at least 3 actionable ETFs or equities (US-listed if possible, or local primary indices) that a retail trader could theoretically buy or short to express this view.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        data = json.loads(response_text)
        return data
        
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Test script locally
    print("Testing geotrade_layer...")
    print(json.dumps(get_geotrade_analysis(), indent=2))
