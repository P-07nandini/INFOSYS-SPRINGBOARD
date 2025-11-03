import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime
from dotenv import load_dotenv
import numpy as np
import cohere
import matplotlib.dates as mdates

# -------------------- LOAD ENV --------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    print("❌ COHERE_API_KEY not found!")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📄 .env file exists: {os.path.exists('.env')}")
    raise ValueError("❌ Please set your COHERE_API_KEY in the .env file.")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

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

# -------------------- TRAIN MODEL (SVR with RBF) --------------------
def train_svr(data):
    """
    Train Support Vector Regressor with RBF (Radial Basis Function) kernel
    
    Parameters:
    - kernel='rbf': Uses Gaussian kernel for non-linear patterns
    - C: Regularization parameter (higher = less regularization)
    - epsilon: Margin of tolerance
    - gamma: Kernel coefficient (how far influence of single training example reaches)
    """
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["avg_sentiment"].values
    
    # Scale features for better SVR performance
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Create and train SVR model
    model = SVR(
        kernel='rbf',       # Radial Basis Function kernel
        C=100.0,            # Regularization parameter
        epsilon=0.1,        # Epsilon in epsilon-SVR model
        gamma='scale'       # Kernel coefficient
    )
    
    model.fit(X_scaled, y_scaled)
    
    # Get predictions and inverse transform
    y_pred_scaled = model.predict(X_scaled)
    data["predicted"] = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Store scalers with model for future predictions
    model.scaler_X = scaler_X
    model.scaler_y = scaler_y
    
    return model, data

# -------------------- FORECAST --------------------
def forecast_future(model, data, days_ahead=5):
    """
    Forecast future sentiment using trained SVR model
    """
    last_index = len(data)
    future_indices = np.arange(last_index, last_index + days_ahead).reshape(-1, 1)
    
    # Scale future indices
    future_indices_scaled = model.scaler_X.transform(future_indices)
    
    # Make predictions
    forecast_scaled = model.predict(future_indices_scaled)
    
    # Inverse transform to original scale
    forecast = model.scaler_y.inverse_transform(forecast_scaled.reshape(-1, 1)).ravel()
    
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
    
    # Plot trend line (SVR fit)
    ax.plot(data["date"], data["predicted"], 
            label="Trend (SVR - RBF Kernel)", linestyle="--", linewidth=2, color="#A23B72")
    
    # Plot forecast
    ax.plot(forecast_df["date"], forecast_df["forecast_sentiment"], 
            label="Forecast", marker="x", linewidth=2, markersize=8, color="#F18F01")
    
    # Title and labels
    ax.set_title("Sentiment Forecast (Support Vector Regressor - RBF Kernel)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Sentiment Score", fontsize=12, fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Show every 2 days
    
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
    plt.savefig("sentiment_trend_forecast_svr.png", dpi=300, bbox_inches='tight')
    print("📊 Graph saved as: sentiment_trend_forecast_svr.png")
    
    plt.show()

# -------------------- LLM SUMMARY --------------------
def generate_summary(data, forecast_df):
    try:
        # Create a more detailed prompt with actual data
        recent_sentiment = data["avg_sentiment"].tail(5).mean()
        forecast_sentiment = forecast_df["forecast_sentiment"].mean()
        trend = "increasing" if forecast_sentiment > recent_sentiment else "decreasing"
        
        prompt = (
            f"You are an AI analyst. Analyze this news sentiment data:\n"
            f"- Recent average sentiment (last 5 days): {recent_sentiment:.3f}\n"
            f"- Forecasted average sentiment (next 5 days): {forecast_sentiment:.3f}\n"
            f"- Trend: {trend}\n"
            f"- Model used: Support Vector Regressor with RBF kernel\n\n"
            f"Provide a concise 2-3 sentence summary of the sentiment trend and what it suggests "
            f"about the news coverage. Use a professional tone."
        )
        
        # Using updated Chat API
        response = co.chat(
            model='command-r',  # Updated model
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
        
        # Fallback summary
        recent_sentiment = data["avg_sentiment"].tail(5).mean()
        forecast_sentiment = forecast_df["forecast_sentiment"].mean()
        trend = "increasing" if forecast_sentiment > recent_sentiment else "decreasing"
        
        fallback_summary = (
            f"News sentiment is {trend}. "
            f"Recent average: {recent_sentiment:.3f}, "
            f"Forecast average: {forecast_sentiment:.3f}. "
            f"SVR model with RBF kernel was used for non-linear pattern detection."
        )
        print(f"\n📰 Basic Summary:\n{fallback_summary}")
        return fallback_summary

# -------------------- MAIN --------------------
def main(forecast_days=5):
    print("="*60)
    print(f"🤖 SENTIMENT FORECAST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Model: Support Vector Regressor (RBF Kernel)")
    print("="*60)
    
    print("\n📊 Loading sentiment data...")
    data = load_data()
    
    # Show data information
    first_date = data["date"].iloc[0]
    last_date = data["date"].iloc[-1]
    print(f"📅 Data range: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
    print(f"📊 Total days of data: {len(data)}")
    print(f"📈 Average sentiment: {data['avg_sentiment'].mean():.4f}")

    print("\n🧮 Training Support Vector Regressor (RBF)...")
    print("   ⚙️ Kernel: Radial Basis Function (RBF)")
    print("   ⚙️ C: 100.0 (regularization)")
    print("   ⚙️ Epsilon: 0.1")
    print("   ⚙️ Gamma: scale")
    model, data = train_svr(data)
    print("✅ SVR model trained successfully!")

    print(f"\n🔮 Forecasting next {forecast_days} days from {last_date.strftime('%Y-%m-%d')}...")
    forecast_df = forecast_future(model, data, days_ahead=forecast_days)
    
    print(f"\n📅 Forecast Results:")
    print("="*50)
    for _, row in forecast_df.iterrows():
        emoji = "📈" if row['forecast_sentiment'] > 0 else "📉"
        print(f"{emoji} {row['date'].strftime('%Y-%m-%d')}: {row['forecast_sentiment']:.4f}")
    print("="*50)

    print("\n📈 Visualizing trend...")
    visualize_trend(data, forecast_df)

    print("\n🧠 Generating AI summary...")
    generate_summary(data, forecast_df)

    print("\n✅ All done! Forecast image and summary ready.")
    print("="*60)

if __name__ == "__main__":
    # You can change forecast_days here
    main(forecast_days=5)