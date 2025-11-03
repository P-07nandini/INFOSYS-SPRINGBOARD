import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime
from dotenv import load_dotenv
import numpy as np
import cohere
import requests
import matplotlib.dates as mdates

# -------------------- LOAD ENV --------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

if not COHERE_API_KEY:
    raise ValueError("❌ Please set your COHERE_API_KEY in the .env file.")

if not SLACK_WEBHOOK_URL:
    print("⚠️ SLACK_WEBHOOK_URL not found. Slack notifications will be disabled.")
    slack_enabled = False
else:
    slack_enabled = True

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

# -------------------- SLACK FUNCTIONS --------------------
def send_slack_message(message):
    """Send message to Slack via webhook"""
    if not slack_enabled:
        print("⚠️ Slack is not configured. Skipping notification.")
        return False
    
    try:
        data = {"text": message}
        response = requests.post(SLACK_WEBHOOK_URL, json=data)
        if response.status_code != 200:
            print(f"❌ Slack Error: {response.text}")
            return False
        print(f"✅ Message sent to Slack")
        return True
    except Exception as e:
        print(f"❌ Slack Error: {e}")
        return False

def send_forecast_to_slack(data, forecast_df, summary):
    """Send detailed forecast information to Slack"""
    
    # Format forecast data
    forecast_text = "\n".join([
        f"   {'📈' if row['forecast_sentiment'] > 0 else '📉'} {row['date'].strftime('%Y-%m-%d')}: {row['forecast_sentiment']:.4f}"
        for _, row in forecast_df.iterrows()
    ])
    
    # Get recent articles with highest sentiment
    try:
        recent_data = pd.read_csv("news_sentiments2_results.csv")
        top_articles = recent_data.nlargest(5, 'score')
        
        articles_text = "\n".join([
            f"   {idx+1}. {row['title'][:80]}... (Score: {row['score']:.3f})"
            for idx, row in top_articles.head(5).iterrows()
        ])
    except:
        articles_text = "   (Article data not available)"
    
    # Calculate trend
    recent_avg = data["avg_sentiment"].tail(5).mean()
    forecast_avg = forecast_df["forecast_sentiment"].mean()
    trend_emoji = "📈" if forecast_avg > recent_avg else "📉"
    trend_text = "increasing" if forecast_avg > recent_avg else "decreasing"
    
    slack_message = f"""
🎯 *Sentiment Analysis Forecast Complete!*

{trend_emoji} *Overall Trend:* {trend_text.upper()}
- Recent Avg (last 5 days): {recent_avg:.4f}
- Forecast Avg (next 5 days): {forecast_avg:.4f}

📊 *5-Day Forecast:*
{forecast_text}

📰 *AI Summary:*
{summary}

🔝 *Top 5 Recent Positive Articles:*
{articles_text}

✅ Full analysis completed successfully!
📁 Check sentiment_trend_forecast_linear.png for visualization
    """
    
    send_slack_message(slack_message)

# -------------------- LOAD DATA --------------------
def load_data(csv_path="news_sentiments2_results.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "publishedAt" not in df.columns:
        raise ValueError("CSV must include 'publishedAt' column from Task 2 output.")

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df = df.dropna(subset=["publishedAt", "score"])
    df["date"] = df["publishedAt"].dt.date
    daily_sentiment = df.groupby("date")["score"].mean().reset_index()
    daily_sentiment.columns = ["date", "avg_sentiment"]
    
    # Convert date to datetime for better plotting
    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
    
    return daily_sentiment

# -------------------- TRAIN MODEL --------------------
def train_linear_regression(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["avg_sentiment"].values

    model_lr = LinearRegression()
    model_lr.fit(X, y)
    data["predicted"] = model_lr.predict(X)
    return model_lr, data

# -------------------- FORECAST --------------------
def forecast_future(model_lr, data, days_ahead=5):
    last_index = len(data)
    future_indices = np.arange(last_index, last_index + days_ahead).reshape(-1, 1)
    forecast = model_lr.predict(future_indices)

    last_date = data["date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    forecast_df = pd.DataFrame({"date": future_dates, "forecast_sentiment": forecast})
    return forecast_df

# -------------------- VISUALIZE (IMPROVED) --------------------
def visualize_trend(data, forecast_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot actual sentiment
    ax.plot(data["date"], data["avg_sentiment"], 
            label="Actual Sentiment", marker="o", linewidth=2, markersize=6, color="#2E86AB")
    
    # Plot trend line
    ax.plot(data["date"], data["predicted"], 
            label="Trend (Linear Regression)", linestyle="--", linewidth=2, color="#A23B72")
    
    # Plot forecast
    ax.plot(forecast_df["date"], forecast_df["forecast_sentiment"], 
            label="Forecast", marker="x", linewidth=2, markersize=8, color="#F18F01")
    
    # Title and labels
    ax.set_title("Sentiment Forecast (Linear Regression Model)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Sentiment Score", fontsize=12, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Show every 5 days
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.6, color='gray')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with higher DPI for better quality
    plt.savefig("sentiment_trend_forecast_linear.png", dpi=300, bbox_inches='tight')
    print("📊 Graph saved as: sentiment_trend_forecast_linear.png")
    
    plt.show()

# -------------------- LLM SUMMARY --------------------
def generate_summary(data, forecast_df):
    try:
        recent_sentiment = data["avg_sentiment"].tail(5).mean()
        forecast_sentiment = forecast_df["forecast_sentiment"].mean()
        trend = "increasing" if forecast_sentiment > recent_sentiment else "decreasing"
        
        prompt = (
            f"You are an AI analyst. Analyze this news sentiment data:\n"
            f"- Recent average sentiment (last 5 days): {recent_sentiment:.3f}\n"
            f"- Forecasted average sentiment (next 5 days): {forecast_sentiment:.3f}\n"
            f"- Trend: {trend}\n\n"
            f"Provide a concise 2-3 sentence summary of the sentiment trend and what it suggests "
            f"about the news coverage. Use a professional tone."
        )
        
        response = co.chat(
            model='command-r',
            message=prompt,
            temperature=0.7,
            max_tokens=150
        )
        
        summary = response.text.strip()
        
        print("\n📰 AI Summary:")
        print(summary)
        return summary
    
    except Exception as e:
        print(f"\n⚠️ Error generating AI summary: {e}")
        recent_sentiment = data["avg_sentiment"].tail(5).mean()
        forecast_sentiment = forecast_df["forecast_sentiment"].mean()
        trend = "increasing" if forecast_sentiment > recent_sentiment else "decreasing"
        
        fallback_summary = (
            f"News sentiment is {trend}. "
            f"Recent average: {recent_sentiment:.3f}, "
            f"Forecast average: {forecast_sentiment:.3f}"
        )
        print(f"\n📰 Basic Summary:\n{fallback_summary}")
        return fallback_summary

# -------------------- MAIN --------------------
def main(forecast_days=5):
    print("="*60)
    print(f"🤖 SENTIMENT FORECAST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    print("\n📊 Loading sentiment data...")
    data = load_data()
    
    # Show data information
    first_date = data["date"].iloc[0]
    last_date = data["date"].iloc[-1]
    print(f"📅 Data range: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"📊 Total days of data: {len(data)}")
    print(f"📈 Average sentiment: {data['avg_sentiment'].mean():.4f}")

    print("\n🧮 Training Linear Regression model...")
    model_lr, data = train_linear_regression(data)

    print(f"\n🔮 Forecasting next {forecast_days} days from {last_date.strftime('%Y-%m-%d')}...")
    forecast_df = forecast_future(model_lr, data, days_ahead=forecast_days)
    
    print(f"\n📅 Forecast Results:")
    print("="*50)
    for _, row in forecast_df.iterrows():
        emoji = "📈" if row['forecast_sentiment'] > 0 else "📉"
        print(f"{emoji} {row['date'].strftime('%Y-%m-%d')}: {row['forecast_sentiment']:.4f}")
    print("="*50)

    print("\n📈 Visualizing trend...")
    visualize_trend(data, forecast_df)

    print("\n🧠 Generating AI summary...")
    summary = generate_summary(data, forecast_df)

    print("\n📨 Sending forecast to Slack...")
    send_forecast_to_slack(data, forecast_df, summary)

    print("\n✅ All done! Forecast image and summary ready.")
    print("="*60)

if __name__ == "__main__":
    # You can change forecast_days here (5, 7, 10, 14, etc.)
    main(forecast_days=5)
