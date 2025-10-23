import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from dotenv import load_dotenv

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Load secrets from environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

if not NEWSAPI_KEY:
    raise ValueError("Please set your NewsAPI key in the .env file")
if not SLACK_WEBHOOK_URL:
    raise ValueError("Please set your Slack webhook URL in the .env file")

# -------------------- TEXT PROCESSING --------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\W", " ", text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def analyze_sentiment(text, sia):
    return sia.polarity_scores(text)["compound"]

# -------------------- FETCH NEWS --------------------
def fetch_news(query, pagesize=100):
    url = (f"https://newsapi.org/v2/everything?q={query}"
           f"&language=en&pageSize={pagesize}&apiKey={NEWSAPI_KEY}")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching news: {response.status_code} - {response.text}")
    return response.json()["articles"]

# -------------------- SLACK --------------------
def send_slack_message(message):
    data = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=data)
    if response.status_code != 200:
        raise Exception(f"Slack notification failed: {response.text}")

# -------------------- MAIN FUNCTION --------------------
def main(query, pagesize=50):
    articles = fetch_news(query, pagesize)
    sia = SentimentIntensityAnalyzer()

    data = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        published = article.get("publishedAt", "")
        combined_text = " ".join([title, description, content])
        clean_text = preprocess_text(combined_text)
        sentiment_score = analyze_sentiment(clean_text, sia)
        sentiment = (
            "Positive" if sentiment_score > 0.05 else
            "Negative" if sentiment_score < -0.05 else
            "Neutral"
        )
        data.append({
            "title": title,
            "description": description,
            "url": article.get("url", ""),
            "publishedAt": published,
            "sentiment": sentiment,
            "score": sentiment_score,
            "processed_text": clean_text
        })

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv("news_sentiments2_results.csv", index=False)
    print("✅ Results saved to news_sentiments2_results.csv")

    # Slack summary
    avg_score = df["score"].mean()
    counts = df["sentiment"].value_counts()
    slack_message = (
        f"📰 Sentiment Analysis for '{query}'\n"
        f"Average Sentiment Score: {avg_score:.3f}\n"
        f"Distribution: {counts.to_dict()}\n"
    )
    send_slack_message(slack_message)
    print("✅ Sentiment summary sent to Slack.")

    # -------------------- VISUALIZATIONS ------------------

    #  Line Chart: Sentiment Trend (by article order)
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["score"], marker="o", linestyle="-", color="blue")
    plt.title("Sentiment Score Trend Across Articles")
    plt.xlabel("Article Index")
    plt.ylabel("Sentiment Score (Compound)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("sentiment_trend_linechart.png")
    plt.show()

    #  Time-based Line Chart (if publishedAt exists)
    if "publishedAt" in df.columns and df["publishedAt"].notna().any():
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df = df.sort_values("publishedAt")

        plt.figure(figsize=(10, 5))
        plt.plot(df["publishedAt"], df["score"], marker="o", color="purple")
        plt.title("Sentiment Trend Over Time")
        plt.xlabel("Published Date")
        plt.ylabel("Sentiment Score")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("sentiment_trend_time.png")
        plt.show()


# -------------------- RUN --------------------
if __name__ == "__main__":
    query = "AI technology"
    pagesize = 100
    main(query, pagesize)
