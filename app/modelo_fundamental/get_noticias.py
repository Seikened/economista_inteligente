import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import yfinance as yf
import pandas as pd
import time

FINNHUB_API_KEY = "d39i039r01qoho9fts30d39i039r01qoho9fts3g" 


@dataclass
class FetchNews:
    tickers: list[str]
    days_back: int = 30
    ticker_news: dict|None = None
    

    def fetch_news(self):
        self.ticker_news = {ticker: {"data": []} for ticker in self.tickers}
        base_url = "https://finnhub.io/api/v1/company-news"
        to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=self.days_back)).strftime('%Y-%m-%d')

        self.error_tickers: list[str] = []
        for ticker in self.tickers:
            print("Esperando 5 segundos")
            time.sleep(5)
            print(f"Fetching news for {ticker}...")
            articles = []
            current_to_date = to_date  # Empezar con la fecha de "hasta" (hace 1 día)
            datetime_from_date = datetime.strptime(from_date, '%Y-%m-%d')
            
            iteration = 0
            response_status = 200
            while True:
                iteration += 1
                print(f"  Iteration {iteration}: from {from_date} to {current_to_date}")
                
                params = {
                    "symbol": ticker,
                    "from": from_date,
                    "to": current_to_date,
                    "token": FINNHUB_API_KEY,
                }
                response = requests.get(base_url, params=params)

                if response.status_code != 200:
                    print(response)
                    print(f"TICKER FALLO: {ticker}")
                    print(f"Error fetching news for {ticker}: {response.status_code}")
                    del self.ticker_news[ticker]
                    self.error_tickers.append(ticker)
                    response_status = response.status_code
                    break
                
                batch_articles = response.json()
                if not batch_articles:
                    print(f"  No more articles found for {ticker}")
                    break
                
                # Agregar artículos al array
                articles.extend(batch_articles)
                print(f"  Fetched {len(batch_articles)} articles, total: {len(articles)}")
                
                # Obtener la fecha del último artículo (más antiguo)
                last_article_timestamp = min(article.get("datetime") for article in batch_articles)
                last_article_date = datetime.fromtimestamp(last_article_timestamp)
                
                print(f"  Oldest article date: {last_article_date.strftime('%Y-%m-%d')}")
                
                # Si ya llegamos a la fecha inicial o antes, terminar
                if last_article_date <= datetime_from_date:
                    print("  Reached initial date, stopping")
                    break
                
                # La nueva fecha "hasta" es la fecha del último artículo menos 1 día
                current_to_date = (last_article_date - timedelta(days=1)).strftime('%Y-%m-%d')
            if response_status != 200:
                continue
            print(f"Total articles fetched for {ticker}: {len(articles)}")
            yfinance_data= yf.download(ticker, start=(datetime.now() - timedelta(days=self.days_back+1)).strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
            if yfinance_data is None or yfinance_data.empty:
                print(f"No stock data found for {ticker}")
                continue
            yfinance_data["rendimiento"]= yfinance_data["Close"].pct_change()

            for article in articles:
                article_date=datetime.fromtimestamp(article.get("datetime")).strftime("%Y-%m-%d")
                try:
                    close_price = round(float(yfinance_data.loc[article_date]["Close"].iloc[0]),4)
                    rendimiento= round(float(yfinance_data.loc[article_date]["rendimiento"].iloc[0]),4)
                except KeyError:
                    # Get closest previous date
                    target = pd.to_datetime(article_date)
                    closest_date = yfinance_data.index[yfinance_data.index <= target][-1]
                    close_price = round(float(yfinance_data.loc[closest_date]["Close"].iloc[0]),2)
                    rendimiento= round(float(yfinance_data.loc[closest_date]["rendimiento"].iloc[0]),4)

                self.ticker_news[ticker]["data"].append({
                    "headline": article.get("headline"),
                    "summary": article.get("summary"),
                    "url": article.get("url"),
                    "source": article.get("source"),
                    "date": article_date,
                    "close": close_price,
                    "performance": rendimiento,
                    "label": "positive" if rendimiento > 0 else "negative"
                    "label": "positive" if rendimiento > 0 else "negative"
                })

        return self.ticker_news
    
    def save_to_json(self, filename="noticias.json"):
        if self.ticker_news is None:
            raise ValueError("No data to save. Please run get_info() first.")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.ticker_news, f, indent=4)

if __name__ == "__main__":


    top_20_tech = [
    #     "NVDA",  # NVIDIA
    #     "MSFT",  # Microsoft
    #     "AAPL",  # Apple
    #     "GOOGL", # Alphabet (Google)
    #     "AMZN",  # Amazon
        "META",  # Meta (Facebook)
        "AVGO",  # Broadcom
        "TSLA",  # Tesla
    #     "TSM",   # Taiwan Semiconductor (TSMC)
    #     "ORCL",  # Oracle
    #     "TCEHY", # Tencent
    #     "NFLX",  # Netflix
    #     "PLTR",  # Palantir
    #     "BABA",  # Alibaba
    #     "ASML",  # ASML Holding
    #     "SAP",   # SAP SE
    #     "CSCO",  # Cisco
    #     "IBM",   # IBM
    #     "AMD"    # Advanced Micro Devices
    ]
    # desde hace 7 años
    news = FetchNews(tickers=top_20_tech, days_back=365*7)
    news.fetch_news()
    news.save_to_json("noticias.json")