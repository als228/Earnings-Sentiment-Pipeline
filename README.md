# Earnings Sentiment Pipeline

A Python pipeline that scores news sentiment around earnings dates for large-cap stocks and tracks short-term price reactions — built to explore whether earnings-period sentiment carries predictive signal.

---

## How It Works

1. **Earnings Calendar** — pulls companies that reported earnings in the last 10 days via Finnhub, filtered to large-caps (revenue > $10B) with reported EPS
2. **News Collection** — fetches news articles from a ±2 day window around each company's earnings date
3. **Sentiment Scoring** — runs each article through [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model pretrained on financial text, classifying sentiment as positive, neutral, or negative
4. **Price Alignment** — pulls closing prices at T-1, T0, and T+2 relative to earnings date via yfinance
5. **Output** — two tables: a detailed per-article table and a summary per-company table with sentiment counts and price data

---

## Why FinBERT

Generic sentiment models struggle with financial language — phrases like "revenue headwinds" or "margin compression" read as neutral to a standard model but are clearly negative in context. FinBERT is trained specifically on financial text and handles this correctly.

---

## Stack

| Component | Library |
|---|---|
| Earnings calendar | Finnhub API |
| News data | yfinance |
| Price data | yfinance |
| Sentiment model | HuggingFace Transformers (FinBERT) |
| Data wrangling | pandas |
| Dashboard | Streamlit |

---

## Setup

```bash
pip install -r requirements.txt
```

Add your Finnhub API key (free at [finnhub.io](https://finnhub.io)) and Huggingface_Hub token to the config section at the top of `earnings_sentiment_pipeline.py`:

```python
FINNHUB_API_KEY = "your_key_here"
API_KEY_HUGGINGFACE_HUB = "your_token_here"
```

Then run:

```bash
python earnings_sentiment_pipeline.py
```

---

## Output

**Detailed table** — one row per article:

| Company | EarningsDate | NewsText | Sentiment | Confidence |
|---|---|---|---|---|
| AAPL | 2025-02-01 | Revenue exceeded expectations... | positive | 0.94 |

**Summary table** — one row per company:

| Company | EarningsDate | PositiveNewsCount | NeutralNewsCount | NegativeNewsCount | Close_T-1 | Close_T0 | Close_T+2 |
|---|---|---|---|---|---|---|---|
| AAPL | 2025-02-01 | 8 | 3 | 1 | 228.50 | 232.10 | 235.40 |

---

## Limitations

- News volume varies significantly by company — smaller large-caps may have few articles in the window
- Sentiment signal is noisy; this is exploratory analysis, not a trading strategy
- FinBERT has a 512 token limit — very long article summaries are truncated
- Price windows around weekends/holidays may have fewer than expected trading days

---

## Requirements

```
finnhub-python
yfinance
transformers
pandas
torch
streamlit
```