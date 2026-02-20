import time
from dotenv import load_dotenv
import os
import datetime as dt
import pandas as pd
import yfinance as yf
import finnhub
import streamlit as st
from transformers import pipeline
from huggingface_hub import login

load_dotenv()

# --- Configuration ---
API_KEY_FINNHUB = os.getenv("FINNHUB_API_KEY")
API_KEY_HUGGINGFACE_HUB = os.getenv("API_KEY_HUGGINGFACE_HUB")
FINBERT_MODEL_PATH = "ProsusAI/finbert"
EARNINGS_LOOKBACK_DAYS = 10
NEWS_WINDOW_DAYS = 2
PRICE_WINDOW_DAYS = 7
MIN_REVENUE_ESTIMATE = 10_000_000_000
SENTIMENT_CONFIDENCE_THRESHOLD = 0.7

# --- Clients & Models ---
sentiment_model = ML_MODEL = pipeline("text-classification", model=FINBERT_MODEL_PATH)
finnhub_client = finnhub.Client(api_key=API_KEY_FINNHUB)
login(token=API_KEY_HUGGINGFACE_HUB)

# --- Date Range ---
end_day = dt.datetime.today().strftime("%Y-%m-%d")
start_day = (dt.datetime.today() - dt.timedelta(days=EARNINGS_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

# --- Earnings Calendar ---
earnings_response = finnhub_client.earnings_calendar(
    _from=start_day, 
    to=end_day, 
    symbol="", 
    international=False
)

earnings_df = pd.DataFrame(earnings_response['earningsCalendar'])
earnings_df = earnings_df.dropna(subset='epsActual')
earnings_df = earnings_df[earnings_df['revenueEstimate'] > MIN_REVENUE_ESTIMATE]
earnings_df['date'] = pd.to_datetime(earnings_df['date'])

# ticker -> earnings date
company_earnings_date_dict = dict(zip(earnings_df['symbol'], earnings_df['date']))

# --- Output Schema ---
detailed_table = pd.DataFrame(
    columns=[
        'Company', 
        'Earnings_Date', 
        'News_Text', 
        'Sentiment', 
        'Confidence'
    ]
)
summary_table = pd.DataFrame(
    columns=[
        'Company', 
        'Earnings_Date', 
        'Sentiment_Score',
        'Close_T-1', 
        'Close_T0', 
        'Close_T+2',
        'Positive_News_Count', 
        'Neutral_News_Count', 
        'Negative_News_Count', 
    ]
)

start_time = time.time()
# --- Main Pipeline ---
for ticker, earnings_date in company_earnings_date_dict.items():
    news_start = (earnings_date - pd.Timedelta(days=NEWS_WINDOW_DAYS)).strftime("%Y-%m-%d")
    news_end = (earnings_date + pd.Timedelta(days=NEWS_WINDOW_DAYS)).strftime("%Y-%m-%d")

    pos_count = neut_count = neg_count = 0

    for article in finnhub_client.company_news(ticker, _from=news_start, to=news_end):
        # prefer summary over headline if available
        content = article['summary'].strip() if article['summary'] and article['summary'].strip() else article['headline']
        output = sentiment_model(content)[0]
        detailed_table.loc[len(detailed_table)] = [ticker, earnings_date, content, output['label'], output['score']]

        if output['score'] > SENTIMENT_CONFIDENCE_THRESHOLD:
            if output['label'] == "positive": pos_count += 1
            elif output['label'] == "neutral": neut_count += 1
            elif output['label'] == "negative": neg_count += 1
    
    # --- Price Data ---
    stock = yf.Ticker(ticker)
    hist = stock.history(
        start=earnings_date-pd.Timedelta(days=7), 
        end=earnings_date+pd.Timedelta(days=7), 
        interval='1d'
    )

    if hist.index.tz is not None: 
        hist.index = hist.index.tz_convert(None) # remove timezones
    
    if not hist.empty:
        close_before = round(hist['Close'].loc[:earnings_date].iloc[-2], 2)
        close_on_day = round(hist['Close'].loc[:earnings_date].iloc[-1], 2)
        close_after = round(hist['Close'].loc[earnings_date:].iloc[2], 2) if len(hist['Close'].loc[earnings_date:]) >= 3 else None
        
        total = pos_count + neut_count + neg_count
        sentiment_score = round((pos_count - neg_count) / total, 2) if total > 0 else None

        summary_table.loc[len(summary_table)] = [
            ticker, 
            earnings_date, 
            sentiment_score,
            close_before,
            close_on_day,
            close_after,
            pos_count, 
            neut_count, 
            neg_count
        ]

# print(f"Pipeline completed in {round(time.time() - start_time, 2)}s")
# print(summary_table)

st.title("Earnings Sentiment Dashboard")

st.subheader("Summary Table")
st.dataframe(summary_table)

st.subheader("Detailed News & Sentiment")
st.dataframe(detailed_table)