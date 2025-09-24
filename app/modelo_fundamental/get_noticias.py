import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

FINNHUB_API_KEY = "d39i039r01qoho9fts30d39i039r01qoho9fts3g" 


@dataclass
class FetchNews:
    tickers: list[str]
    days_back: int = 30
    ticker_news: dict = None


    def get_info(self):
        self.ticker_news = {ticker: {"data": []} for ticker in self.tickers}
        self.fetch_news()
        return self.ticker_news

    def fetch_news(self):
        base_url = "https://finnhub.io/api/v1/company-news"
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=self.days_back)).strftime('%Y-%m-%d')

        for ticker in self.tickers:
            params = {
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY,
            }

            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                print(f"Error fetching news for {ticker}: {response.status_code}")
                continue

            articles = response.json()

            for article in articles:
                self.ticker_news[ticker]["data"].append({
                    "headline": article.get("headline"),
                    "summary": article.get("summary"),
                    "url": article.get("url"),
                    "source": article.get("source"),
                    "date": datetime.fromtimestamp(article.get("datetime")).strftime("%Y-%m-%d"),
                })

        return self.ticker_news


# --- Ejemplo ---

if __name__ == "__main__":
    news = FetchNews(tickers=["AAPL", "TSLA", "MSFT"], days_back=180).get_info()
    print(f"TSLA tiene {len(news['TSLA']['data'])} artículos.")
    for item in news['TSLA']['data']:
        print(item["date"], "-", item["headline"])
    with open ("noticias.json", "w", encoding="utf-8") as f:
        json.dump(news, f, indent=4)
