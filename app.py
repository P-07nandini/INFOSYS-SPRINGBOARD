import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

# -----------------------
# Slack webhook from environment
# -----------------------
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    layout="wide",
    page_title="SENTIMENT NEWS MONITOR",
    page_icon="📰"
)

# -----------------------
# Mock news data
# -----------------------
news_data = [
    {"id": 1, "title": "New Wave Of Innovation - Converge", "score": 0.8, "sentiment": "positive", "source": "TechNews", "date": "2025-11-11", "url": "https://example.com/innovation"},
    {"id": 2, "title": "Request broke investment", "score": -0.8, "sentiment": "negative", "source": "Finance", "date": "2025-11-11", "url": "https://finance.com/article1"},
    {"id": 3, "title": "Returns", "score": -0.1, "sentiment": "neutral", "source": "Markets", "date": "2025-11-10", "url": "https://www.fortune.com/article"},
    {"id": 4, "title": "Until its best Q4 results ever", "score": 0.9, "sentiment": "positive", "source": "Business", "date": "2025-11-10", "url": "https://business.com/results"},
    {"id": 5, "title": "Major tech company announces layoffs", "score": -0.7, "sentiment": "negative", "source": "TechCrunch", "date": "2025-11-09", "url": "https://techcrunch.com/layoffs"},
    {"id": 6, "title": "Market shows steady growth", "score": 0.3, "sentiment": "positive", "source": "Bloomberg", "date": "2025-11-09", "url": "https://bloomberg.com/growth"},
    {"id": 7, "title": "Regulatory concerns emerge", "score": -0.4, "sentiment": "negative", "source": "Reuters", "date": "2025-11-08", "url": "https://reuters.com/regulatory"},
    {"id": 8, "title": "Innovation breakthrough in AI", "score": 0.85, "sentiment": "positive", "source": "Nature", "date": "2025-11-08", "url": "https://nature.com/ai-breakthrough"},
]

df = pd.DataFrame(news_data)
df['date'] = pd.to_datetime(df['date']).dt.date

# -----------------------
# Sidebar filters
# -----------------------
st.sidebar.title("News Monitor")

# Sentiment selection
sentiments = ["positive", "negative", "neutral"]
selected_sentiments = [s for s in sentiments if st.sidebar.checkbox(s.capitalize(), value=True)]

# Score filters
min_score = st.sidebar.slider("Score min", -1.0, 1.0, -1.0, 0.1)
max_score = st.sidebar.slider("Score max", -1.0, 1.0, 1.0, 0.1)

if min_score > max_score:
    st.sidebar.error("Min cannot be greater than Max")

# -----------------------
# Slack Configuration in Sidebar
# -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Slack Configuration")

# Allow users to input webhook URL if not in environment
if not SLACK_WEBHOOK_URL:
    webhook_input = st.sidebar.text_input(
        "Slack Webhook URL",
        type="password",
        help="Enter your Slack Incoming Webhook URL"
    )
    if webhook_input:
        SLACK_WEBHOOK_URL = webhook_input
        st.sidebar.success("✅ Webhook configured!")
    else:
        st.sidebar.warning("⚠️ No webhook URL configured")
else:
    st.sidebar.success("✅ Webhook loaded from environment")
    # Show masked webhook
    masked = SLACK_WEBHOOK_URL[:30] + "..." if len(SLACK_WEBHOOK_URL) > 30 else SLACK_WEBHOOK_URL
    st.sidebar.info(f"URL: {masked}")

# -----------------------
# Filter function
# -----------------------
def apply_filters(df, sentiments_list, min_s, max_s):
    mask = (df['score'] >= min_s) & (df['score'] <= max_s)
    if sentiments_list:
        mask &= df['sentiment'].isin(sentiments_list)
    return df[mask]

filtered = apply_filters(df, selected_sentiments, min_score, max_score)

# -----------------------
# Header metrics
# -----------------------
col1, col2, col3 = st.columns(3)
counts = filtered['sentiment'].value_counts().to_dict()
col1.metric("Positive", counts.get('positive', 0))
col2.metric("Negative", counts.get('negative', 0))
col3.metric("Neutral", counts.get('neutral', 0))

# -----------------------
# Improved Slack alert function
# -----------------------
def send_slack_alert(webhook_url, items):
    """Send news articles to Slack via webhook"""
    if not webhook_url:
        return {"ok": False, "error": "No webhook URL configured. Please add one in the sidebar."}
    
    if not items:
        return {"ok": False, "error": "No articles to send"}
    
    # Format the message
    sentiment_counts = pd.DataFrame(items)['sentiment'].value_counts().to_dict()
    
    text = f"📰 *News Monitor Alert*\n"
    text += f"Total Articles: {len(items)}\n"
    text += f"• Positive: {sentiment_counts.get('positive', 0)}\n"
    text += f"• Negative: {sentiment_counts.get('negative', 0)}\n"
    text += f"• Neutral: {sentiment_counts.get('neutral', 0)}\n\n"
    text += f"*Top Articles:*\n"
    
    # Send up to 10 articles in the initial message
    for i, item in enumerate(items[:10], 1):
        emoji = "🟢" if item['sentiment'] == 'positive' else "🔴" if item['sentiment'] == 'negative' else "⚪"
        text += f"{emoji} *{item['title']}*\n"
        text += f"   Score: {item['score']:+.2f} | Source: {item['source']}\n"
        text += f"   <{item['url']}|Read More>\n\n"
    
    if len(items) > 10:
        text += f"\n_... and {len(items) - 10} more articles_"
    
    payload = {"text": text}
    
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        
        if resp.ok:
            return {"ok": True, "message": "Alert sent successfully!"}
        else:
            return {
                "ok": False, 
                "error": f"Slack API error (Status {resp.status_code}): {resp.text[:200]}"
            }
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Request timed out. Please check your connection."}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"ok": False, "error": f"Unexpected error: {str(e)}"}

# -----------------------
# Slack Alert Section
# -----------------------
st.markdown("---")
st.header("📬 Send Slack Alerts")

if len(filtered) == 0:
    st.warning("⚠️ No articles match your current filters. Adjust filters to see articles.")
else:
    st.write(f"📊 **{len(filtered)} news articles** available with applied filters")
    
    col_a, col_b = st.columns([1, 3])
    
    with col_a:
        send_button_disabled = not SLACK_WEBHOOK_URL
        if st.button("Send Slack Alert 🚀", disabled=send_button_disabled, type="primary"):
            with st.spinner("Sending alert..."):
                result = send_slack_alert(SLACK_WEBHOOK_URL, filtered.to_dict('records'))
                
                if result.get("ok"):
                    st.success(f"✅ {result.get('message', 'Alert sent successfully!')}")
                else:
                    st.error(f"❌ Failed to send alert: {result.get('error', 'Unknown error')}")
    
    with col_b:
        if not SLACK_WEBHOOK_URL:
            st.info("💡 **Configure your Slack webhook URL in the sidebar to enable alerts**")

# -----------------------
# Charts
# -----------------------
st.markdown("---")

# Pie chart
sent_count = filtered['sentiment'].value_counts().reindex(['positive', 'negative', 'neutral']).fillna(0)
pie_df = pd.DataFrame({
    "sentiment": ['Positive','Negative','Neutral'],
    "value": [sent_count.get('positive',0), sent_count.get('negative',0), sent_count.get('neutral',0)]
})
fig_pie = px.pie(
    pie_df, names='sentiment', values='value', 
    color='sentiment', 
    color_discrete_map={'Positive':'#10b981','Negative':'#ef4444','Neutral':'#6b7280'},
    hole=0.35
)
fig_pie.update_traces(textinfo='label+value')

# Daily breakdown
daily = filtered.groupby('date').agg(
    positive = ('sentiment', lambda s: (s=='positive').sum()),
    negative = ('sentiment', lambda s: (s=='negative').sum()),
    neutral = ('sentiment', lambda s: (s=='neutral').sum()),
    avgScore = ('score', 'mean'),
    count = ('score', 'count')
).reset_index().sort_values('date')

fig_bar = go.Figure()
fig_bar.add_bar(name='Positive', x=daily['date'], y=daily['positive'], marker_color='#10b981')
fig_bar.add_bar(name='Negative', x=daily['date'], y=daily['negative'], marker_color='#ef4444')
fig_bar.add_bar(name='Neutral', x=daily['date'], y=daily['neutral'], marker_color='#6b7280')
fig_bar.update_layout(barmode='group', xaxis_title='Date', yaxis_title='Count', template='plotly_dark')

# Avg score trend line
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=daily['date'], y=daily['avgScore'], mode='lines+markers',
    line=dict(color='#8b5cf6', width=3), marker=dict(size=8)
))
fig_line.update_layout(yaxis=dict(range=[-1,1]), xaxis_title='Date', yaxis_title='Average Score', template='plotly_dark')

# Layout
c1, c2 = st.columns(2)
with c1:
    st.subheader("Sentiment Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
with c2:
    st.subheader("Daily Sentiment Breakdown")
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Average Sentiment Score Trend")
st.plotly_chart(fig_line, use_container_width=True)

# -----------------------
# News list
# -----------------------
st.markdown("---")
st.subheader("📋 Filtered News Articles")

if len(filtered) == 0:
    st.info("No articles match your filters. Try adjusting the sentiment or score filters.")
else:
    for _, row in filtered.sort_values('date', ascending=False).iterrows():
        cols = st.columns([0.85, 0.15])
        with cols[0]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"*{row['source']} — {row['date'].strftime('%Y-%m-%d')}*")
            st.markdown(f"[View Article]({row['url']})")
        with cols[1]:
            color = {"positive":"#10b981","negative":"#ef4444","neutral":"#6b7280"}.get(row['sentiment'], "#6b7280")
            st.markdown(
                f"<div style='text-align:right'>"
                f"<span style='background:{color};padding:6px;border-radius:8px;color:white;font-weight:600'>{row['sentiment'].upper()}</span>"
                f"<div style='font-size:12px;color:#9CA3AF;margin-top:4px'>{row['score']:+.2f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("---")