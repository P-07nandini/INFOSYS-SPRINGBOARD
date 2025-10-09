import os
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# ============================== #
#   Your NewsAPI Key (Required)  #
# ============================== #
NEWSAPI_KEY = "b658661eea3145bea4a8d89b96a760a0"

if not NEWSAPI_KEY:
    raise ValueError("Please set your NewsAPI key before running!")

# ============================== #
#      Text Preprocessing        #
# ============================== #
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # remove links
    text = re.sub(r"\W", " ", text)  # remove non-alphanumeric chars
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ============================== #
#      Sentiment Analysis        #
# ============================== #
def analyze_sentiment(text, sia):
    return sia.polarity_scores(text)["compound"]

# ============================== #
#      News Fetching             #
# ============================== #
def fetch_news(query, pagesize=100):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language=en&pageSize={pagesize}&apiKey={NEWSAPI_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching news: {response.status_code} - {response.text}")
    return response.json()["articles"]

# ============================== #
#       Main Program             #
# ============================== #
def main(query, pagesize=100):
    articles = fetch_news(query, pagesize)
    sia = SentimentIntensityAnalyzer()

    data = []
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
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
            "sentiment": sentiment,
            "score": sentiment_score,
            "processed_text": clean_text
        })

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv("news_sentiment_results.csv", index=False)
    print(" Results saved to news_sentiment_results.csv")

    # Plot sentiment distribution
    plt.figure(figsize=(6, 4))
    df["sentiment"].value_counts().plot(kind="bar", color=["green", "red", "gray"])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png")
    plt.show()

    # Plot top words
    all_words = " ".join(df["processed_text"])
    word_freq = pd.Series(all_words.split()).value_counts().head(20)
    plt.figure(figsize=(8, 5))
    word_freq.plot(kind="bar")
    plt.title("Top Words in News Articles")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("top_words.png")
    plt.show()

    # WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("news_wordcloud.png")
    plt.show()

if __name__ == "__main__":
    query = "AI technology"
    pagesize = 50
    main(query, pagesize)
