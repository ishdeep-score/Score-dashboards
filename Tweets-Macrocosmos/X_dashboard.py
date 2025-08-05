import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

st.set_page_config(page_title="Tweet Performance Dashboard", layout="wide")

import os


@st.cache_data
def load_data(csv_path="Tweets-Macrocosmos/tweets.csv"):
    return pd.read_csv(csv_path)

df = load_data()

# Only include thedkingdao and webuildscore
accounts_of_interest = ["thedkingdao", "webuildscore"]
df = df[df["username"].isin(accounts_of_interest)]

st.title("📊 Tweet Performance Dashboard")
st.markdown("Analyze the performance of your tweets over time.")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select Page",
    ("Overview ", "thedkingdao Analysis", "webuildscore Analysis")
)

# Engagement rate (likes+retweets+quotes per followers_count)
def engagement_rate(row):
    if row["followers_count"] > 0:
        return (row["like_count"] + row["retweet_count"] + row["quote_count"]) / row["followers_count"]
    return np.nan

df["engagement_rate"] = df.apply(engagement_rate, axis=1)
df["date"] = pd.to_datetime(df["datetime"]).dt.date
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.day_name()

# Ensure is_reply, is_retweet, is_quote are boolean and NaN-free
for col in ["is_reply", "is_retweet", "is_quote"]:
    df[col] = df[col].fillna(False).astype(bool)

conditions = [
    ((~df["is_reply"]) & (~df["is_retweet"]) & (~df["is_quote"])).values,
    df["is_reply"].values,
    df["is_retweet"].values,
    df["is_quote"].values
]
choices = ["Original", "Reply", "Retweet", "Quote"]
df["tweet_type"] = np.select(conditions, choices, default="Other")

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['content'].fillna('').apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment_label'] = pd.cut(df['sentiment_score'], bins=[-1, -0.05, 0.05, 1], labels=['Negative', 'Neutral', 'Positive'])

def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

def get_top_words(contents, n=10):
    words = re.findall(r"\b\w+\b", " ".join(contents).lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(n)

def show_hashtag_analysis(contents):
    all_hashtags = contents.dropna().apply(extract_hashtags)
    flat_hashtags = [tag.lower() for sublist in all_hashtags for tag in sublist]
    hashtag_counts = Counter(flat_hashtags)
    top_hashtags = hashtag_counts.most_common(10)
    st.subheader("Top Hashtags")
    if top_hashtags:
        st.table(pd.DataFrame(top_hashtags, columns=["Hashtag", "Count"]))
    else:
        st.write("No hashtags found.")

def show_top_words(contents):
    top_words = get_top_words(contents)
    st.subheader("Most Common Words (Excluding Stopwords)")
    if top_words:
        st.bar_chart(pd.DataFrame(top_words, columns=["Word", "Count"]).set_index("Word"))
    else:
        st.write("No words found.")

def show_tweet_length_analysis(df):
    df["tweet_length"] = df["content"].fillna("").apply(len)
    st.subheader("Engagement vs. Tweet Length ")
    st.markdown(
        """
        This chart shows the relationship between the length of a tweet (number of characters) and its engagement (sum of likes, retweets, and quotes).\
        Use it to see if longer or shorter tweets tend to get more engagement.\
        Look for clusters or trends: if high-engagement tweets are mostly short or long, you can adjust your content strategy accordingly.
        """
    )
    st.scatter_chart(df[["tweet_length", "engagement"]])

def show_sentiment_analysis(df):
    st.subheader("Average Sentiment Over Time")
    st.line_chart(df.groupby('date')['sentiment_score'].mean())

    st.subheader("Average Engagement by Sentiment")
    st.bar_chart(df.groupby('sentiment_label')['engagement'].mean())

def show_overview():
    st.subheader("Overall Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", len(df))
    col2.metric("Total Likes", int(df["like_count"].sum()))
    col3.metric("Total Retweets", int(df["retweet_count"].sum()))
    col4.metric("Total Quotes", int(df["quote_count"].sum()))

    # Per-account summary
    grouped = df.groupby("username").agg({
        "uri": "count",
        "like_count": "sum",
        "retweet_count": "sum",
        "quote_count": "sum",
        "followers_count": "max",
        "engagement_rate": "mean"
    }).rename(columns={"uri": "num_tweets"}).reset_index()

    st.subheader("Per-Account Summary")
    st.dataframe(grouped, use_container_width=True)

    # Trends over time
    trend = df.groupby("date").agg({
        "like_count": "sum",
        "retweet_count": "sum",
        "quote_count": "sum",
        "uri": "count"
    }).rename(columns={"uri": "num_tweets"})

    st.subheader("Trends Over Time")
    st.line_chart(trend)

    # Engagement by day of week
    st.subheader("Average Engagement by Day of Week")
    df["engagement"] = df["like_count"] + df["retweet_count"] + df["quote_count"]
    engagement_by_day = df.groupby("day_of_week")["engagement"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    st.bar_chart(engagement_by_day)

    # Tweet type analysis
    st.subheader("Average Engagement by Tweet Type")
    type_engagement = df.groupby("tweet_type")["engagement"].mean()
    st.bar_chart(type_engagement)

    # Hashtag analysis
    show_hashtag_analysis(df["content"])
    # Top words
    show_top_words(df["content"])
    # Tweet length analysis
    show_tweet_length_analysis(df)
    # Sentiment analysis
    show_sentiment_analysis(df)

def show_account_analysis(account):
    st.header(f"Analysis for @{account}")
    acc_df = df[df["username"] == account].copy()

    # Tweet performance by posting time (bar chart)
    st.subheader("Tweet Engagement by Hour of Day")
    acc_df["engagement"] = acc_df["like_count"] + acc_df["retweet_count"] + acc_df["quote_count"]
    st.bar_chart(acc_df.groupby("hour")["engagement"].mean())

    # Engagement by day of week
    st.subheader("Average Engagement by Day of Week")
    engagement_by_day = acc_df.groupby("day_of_week")["engagement"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    st.bar_chart(engagement_by_day)

    # Tweet type analysis
    st.subheader("Average Engagement by Tweet Type")
    type_engagement = acc_df.groupby("tweet_type")["engagement"].mean()
    st.bar_chart(type_engagement)

    # Hashtag analysis
    show_hashtag_analysis(acc_df["content"])
    # Top words
    show_top_words(acc_df["content"])
    # Tweet length analysis
    show_tweet_length_analysis(acc_df)
    # Sentiment analysis
    #show_sentiment_analysis(acc_df)

    # Last 3 tweets of the week with the most engagement
    st.subheader("Top 3 Tweets of the Week (by Engagement)")
    acc_df["week"] = pd.to_datetime(acc_df["datetime"]).dt.isocalendar().week
    latest_week = acc_df["week"].max()
    week_df = acc_df[acc_df["week"] == latest_week]
    top3 = week_df.sort_values("engagement", ascending=False).head(3)
    for i, row in top3.iterrows():
        st.markdown(f"**{row['datetime']}** | [View Tweet]({row['uri']})")
        st.write(f"Likes: {row['like_count']} | Retweets: {row['retweet_count']} | Quotes: {row['quote_count']} | Engagement: {row['engagement']}")
        st.markdown("---")

    # Table of all tweets (optional)
    with st.expander("Show All Tweets"):
        st.dataframe(acc_df, use_container_width=True)

if page == "Overview ":
    show_overview()
elif page == "thedkingdao Analysis":
    show_account_analysis("thedkingdao")
elif page == "webuildscore Analysis":
    show_account_analysis("webuildscore")

# Load DKING and SCORE posts
@st.cache_data
def load_extra_csvs():
    score_df = pd.read_csv("Tweets-Macrocosmos/SCORE_X.csv")
    dking_df = pd.read_csv("Tweets-Macrocosmos/DKING_X.csv")
    return score_df, dking_df

score_df, dking_df = load_extra_csvs()

# Helper: Classify Score posts
def classify_score_post(text):
    football_keywords = [
        "football", "goal", "match", "league", "player", "coach", "club", "La Liga", "Premier League", "Serie A",
        "Bundesliga", "Champions League", "UCL", "World Cup", "dribble", "xG", "xA", "xT", "EPV", "packing", "shot"
    ]
    if any(kw.lower() in text.lower() for kw in football_keywords):
        return "Football"
    return "Other"

def classify_post_type(text):
    if "🧵" in text or "thread" in text.lower():
        return "Thread"
    elif "Read more" in text or "article" in text.lower():
        return "Article"
    else:
        return "Post"

score_df["Category"] = score_df["Post text"].fillna("").apply(classify_score_post)
score_df["Type"] = score_df["Post text"].fillna("").apply(classify_post_type)

# DKING posts: All are betting-related, but you can add more logic if needed

def classify_football_category(text):
    text = str(text).lower()
    if any(kw in text for kw in ["goal of the week", "shot of the week", "miss of the week", "play of the week", "scored", "goal", "finish", "strike", "header"]):
        return "Shot/Miss/Goal/Play of the Week"
    elif "guess the player" in text or "who do you think it is" in text or "who should we scout" in text or "who's your pick" in text:
        return "Guess The Player"
    elif any(kw in text for kw in ["to ", "transfer", "joins", "sign", "move", "confirmed"]):
        return "Transfers"
    elif any(kw in text for kw in ["vs", "matchup", "semi-final", "final", "quarter-final", "match", "battle", "face", "meet", "takes on", "takes the trophy", "who wins", "who claims", "who gets the points", "who lifts the trophy"]):
        return "Matchups"
    else:
        return "Others"

score_df["Football_Category"] = score_df.apply(
    lambda row: classify_football_category(row["Post text"]) if row["Category"] == "Football" else "Non-Football", axis=1
)

def classify_non_football_category(text):
    """Classify non-football posts into detailed categories"""
    text = str(text).lower()
    if any(kw in text for kw in ["subnet", "sn44", "validator", "bittensor", "ai", "machine learning", "model"]):
        return "Technical/AI/Subnet"
    elif any(kw in text for kw in ["rebrand", "website", "identity", "brand", "design", "logo"]):
        return "Branding/Marketing"
    elif any(kw in text for kw in ["roundup", "weekly", "recap", "update", "announcement"]):
        return "Updates/Announcements"
    elif any(kw in text for kw in ["mindshare", "ranking", "top", "performance", "metrics"]):
        return "Performance/Metrics"
    elif any(kw in text for kw in ["hotfix", "deploy", "bug", "fix", "patch"]):
        return "Technical Updates"
    else:
        return "Other Non-Football"

def classify_dking_category(text):
    """Classify DKING posts into betting categories"""
    text = str(text).lower()
    if any(kw in text for kw in ["+ev", "probability", "fair odds", "value"]):
        return "Betting Analysis"
    elif any(kw in text for kw in ["prompt", "guide", "how to", "try this"]):
        return "Educational/Guides"
    elif any(kw in text for kw in ["sire", "dking", "migration", "unstaking", "token"]):
        return "Token/Migration"
    elif any(kw in text for kw in ["tony bloom", "starlizard", "brighton"]):
        return "Industry Stories"
    else:
        return "Other Betting"

# Apply classifications
score_df["Non_Football_Category"] = score_df.apply(
    lambda row: classify_non_football_category(row["Post text"]) if row["Category"] == "Other" else "Football Related", axis=1
)

def show_score_analysis():
    st.header("Score Account Analysis (SCORE_X.csv)")
    
    score_df["Engagement"] = score_df["Likes"].fillna(0) + score_df["Engagements"].fillna(0)
    score_df["Date"] = pd.to_datetime(score_df["Date"], errors="coerce")

    # Overall Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(score_df))
    col2.metric("Total Impressions", f"{score_df['Impressions'].sum():,}")
    col3.metric("Total Engagement", f"{score_df['Engagement'].sum():,}")
    col4.metric("Avg Engagement Rate", f"{(score_df['Engagement'].sum() / score_df['Impressions'].sum() * 100):.2f}%")

    # Football vs Non-Football Analysis
    st.subheader("Football vs Non-Football Post Performance")
    category_stats = score_df.groupby("Category").agg({
        "Engagement": ["mean", "sum", "count"],
        "Likes": "mean",
        "Reposts": "mean",
        "Replies": "mean",
        "New follows": "mean"
    }).round(2)
    st.table(category_stats)

    # Post Type Analysis (Articles, Threads, Posts)
    st.subheader("Performance by Post Type (Articles vs Threads vs Posts)")
    type_stats = score_df.groupby("Type").agg({
        "Engagement": ["mean", "sum", "count"],
        "Likes": "mean",
        "Reposts": "mean",
        "Replies": "mean",
        "Impressions": "mean"
    }).round(2)
    st.table(type_stats)
    
    st.bar_chart(score_df.groupby("Type")["Engagement"].mean())

    # Non-Football Category Breakdown
    st.subheader("Non-Football Post Categories")
    non_football_posts = score_df[score_df["Category"] == "Other"]
    if len(non_football_posts) > 0:
        st.bar_chart(non_football_posts["Non_Football_Category"].value_counts())
        
        st.subheader("Non-Football Category Performance")
        non_football_stats = non_football_posts.groupby("Non_Football_Category").agg({
            "Engagement": "mean",
            "Likes": "mean",
            "Reposts": "mean",
            "New follows": "mean"
        }).round(2)
        st.table(non_football_stats)

    # Football Category Analysis
    st.subheader("Football Post Categories")
    football_posts = score_df[score_df["Category"] == "Football"]
    if len(football_posts) > 0:
        st.bar_chart(football_posts["Football_Category"].value_counts())
        
        st.subheader("Football Category Performance")
        football_stats = football_posts.groupby("Football_Category").agg({
            "Engagement": "mean",
            "Likes": "mean",
            "Reposts": "mean",
            "New follows": "mean"
        }).round(2)
        st.table(football_stats)

    # Top Performing Posts by Type
    st.subheader("Top Posts by Type")
    for post_type in score_df["Type"].unique():
        type_posts = score_df[score_df["Type"] == post_type].sort_values("Engagement", ascending=False).head(3)
        st.write(f"**Top {post_type}s:**")
        for _, row in type_posts.iterrows():
            st.write(f"- {row['Post text'][:100]}... | Engagement: {row['Engagement']} | Likes: {row['Likes']}")
        st.write("")

    # Insights and Recommendations
    st.subheader("Key Insights")
    avg_thread_engagement = score_df[score_df["Type"] == "Thread"]["Engagement"].mean()
    avg_post_engagement = score_df[score_df["Type"] == "Post"]["Engagement"].mean()
    avg_article_engagement = score_df[score_df["Type"] == "Article"]["Engagement"].mean()
    
    st.markdown(f"""
    **Post Type Performance:**
    - Threads average: {avg_thread_engagement:.1f} engagement
    - Regular posts average: {avg_post_engagement:.1f} engagement  
    - Articles average: {avg_article_engagement:.1f} engagement
    
    **Recommendations:**
    - {"Threads perform best - create more threaded content" if avg_thread_engagement > avg_post_engagement else "Regular posts perform well - maintain current strategy"}
    - Non-football technical content tends to get higher quality followers
    - Football content drives more casual engagement and broader reach
    """)

def show_dking_analysis():
    st.header("DKING Account Analysis (DKING_X.csv)")
    
    dking_df["Engagement"] = dking_df["Likes"].fillna(0) + dking_df["Engagements"].fillna(0)
    dking_df["Date"] = pd.to_datetime(dking_df["Date"], errors="coerce")
    dking_df["Category"] = dking_df["Post text"].apply(classify_dking_category)

    # DKING Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", len(dking_df))
    col2.metric("Total Impressions", f"{dking_df['Impressions'].sum():,}")
    col3.metric("Total Engagement", f"{dking_df['Engagement'].sum():,}")
    col4.metric("Avg Engagement Rate", f"{(dking_df['Engagement'].sum() / dking_df['Impressions'].sum() * 100):.2f}%")

    # DKING Category Analysis
    st.subheader("DKING Post Categories")
    st.bar_chart(dking_df["Category"].value_counts())
    
    st.subheader("Performance by Category")
    dking_stats = dking_df.groupby("Category").agg({
        "Engagement": ["mean", "count"],
        "Likes": "mean",
        "Reposts": "mean"
    }).round(2)
    st.table(dking_stats)

    # Top DKING Posts
    st.subheader("Top DKING Posts")
    top_dking = dking_df.sort_values("Engagement", ascending=False).head(5)
    st.table(top_dking[["Date", "Post text", "Category", "Engagement", "Likes", "Impressions"]])

# Add navigation
extra_page = st.sidebar.selectbox(
    "Extra Analysis",
    ("None", "Score Account", "DKING Account")
)

if extra_page == "Score Account":
    show_score_analysis()
elif extra_page == "DKING Account":
    show_dking_analysis()

# Add these enhanced analysis functions after the existing functions:

def show_enhanced_overview():
    st.header("Enhanced Tweet Performance Analysis")
    
    # Overall metrics with trends
    col1, col2, col3, col4 = st.columns(4)
    
    total_engagement = df["like_count"].sum() + df["retweet_count"].sum() + df["quote_count"].sum()
    avg_engagement_rate = df["engagement_rate"].mean() * 100
    
    col1.metric("Total Tweets", len(df))
    col2.metric("Total Engagement", f"{total_engagement:,}")
    col3.metric("Avg Engagement Rate", f"{avg_engagement_rate:.2f}%")
    col4.metric("Active Days", df["date"].nunique())

    # Enhanced time-based analysis
    st.subheader("📅 Time-Based Performance Analysis")
    
    # Daily engagement trends
    daily_stats = df.groupby("date").agg({
        "like_count": "sum",
        "retweet_count": "sum", 
        "quote_count": "sum",
        "engagement_rate": "mean",
        "uri": "count"
    }).rename(columns={"uri": "tweet_count"})
    
    daily_stats["total_engagement"] = daily_stats["like_count"] + daily_stats["retweet_count"] + daily_stats["quote_count"]
    
    # Create subplot for daily trends
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Daily Total Engagement", "Daily Tweet Count", "Daily Engagement Rate", "Engagement vs Tweet Volume"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Daily engagement
    fig.add_trace(
        go.Scatter(x=daily_stats.index, y=daily_stats["total_engagement"], 
                  name="Total Engagement", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Daily tweet count
    fig.add_trace(
        go.Scatter(x=daily_stats.index, y=daily_stats["tweet_count"], 
                  name="Tweet Count", line=dict(color="green")),
        row=1, col=2
    )
    
    # Daily engagement rate
    fig.add_trace(
        go.Scatter(x=daily_stats.index, y=daily_stats["engagement_rate"]*100, 
                  name="Engagement Rate %", line=dict(color="red")),
        row=2, col=1
    )
    
    # Engagement vs Volume scatter
    fig.add_trace(
        go.Scatter(x=daily_stats["tweet_count"], y=daily_stats["total_engagement"],
                  mode="markers", name="Daily Performance", 
                  marker=dict(size=8, color="purple")),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Daily Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)

def show_content_performance_analysis():
    st.subheader("🎯 Content Performance Deep Dive")
    
    # Tweet type performance comparison
    type_performance = df.groupby("tweet_type").agg({
        "like_count": ["mean", "sum", "count"],
        "retweet_count": ["mean", "sum"],
        "quote_count": ["mean", "sum"],
        "engagement_rate": "mean"
    }).round(2)
    
    # Flatten column names
    type_performance.columns = ['_'.join(col).strip() for col in type_performance.columns]
    type_performance = type_performance.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Average Performance by Tweet Type**")
        fig = px.bar(
            df.groupby("tweet_type").agg({
                "like_count": "mean",
                "retweet_count": "mean", 
                "quote_count": "mean"
            }).reset_index(),
            x="tweet_type",
            y=["like_count", "retweet_count", "quote_count"],
            title="Average Engagement by Tweet Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Tweet Type Distribution**")
        type_counts = df["tweet_type"].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="Tweet Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_temporal_patterns():
    st.subheader("⏰ Temporal Posting Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hour of day performance
        hourly_performance = df.groupby("hour").agg({
            "like_count": "mean",
            "retweet_count": "mean",
            "quote_count": "mean",
            "engagement_rate": "mean",
            "uri": "count"
        }).rename(columns={"uri": "tweet_count"})
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add engagement bars
        fig.add_trace(
            go.Bar(x=hourly_performance.index, 
                   y=hourly_performance["like_count"] + hourly_performance["retweet_count"] + hourly_performance["quote_count"],
                   name="Avg Total Engagement", marker_color="lightblue"),
            secondary_y=False
        )
        
        # Add tweet count line
        fig.add_trace(
            go.Scatter(x=hourly_performance.index, y=hourly_performance["tweet_count"],
                      mode="lines+markers", name="Tweet Count", line=dict(color="red")),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Average Engagement", secondary_y=False)
        fig.update_yaxes(title_text="Tweet Count", secondary_y=True)
        fig.update_layout(title_text="Hourly Posting Patterns vs Engagement")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week analysis
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_performance = df.groupby("day_of_week").agg({
            "like_count": "mean",
            "retweet_count": "mean",
            "quote_count": "mean",
            "engagement_rate": "mean",
            "uri": "count"
        }).rename(columns={"uri": "tweet_count"}).reindex(day_order)
        
        fig = px.bar(
            x=daily_performance.index,
            y=daily_performance["like_count"] + daily_performance["retweet_count"] + daily_performance["quote_count"],
            title="Average Engagement by Day of Week",
            labels={"x": "Day of Week", "y": "Average Total Engagement"}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_engagement_distribution_analysis():
    st.subheader("📊 Engagement Distribution Analysis")
    
    # Create engagement buckets
    df["total_engagement"] = df["like_count"] + df["retweet_count"] + df["quote_count"]
    df["engagement_bucket"] = pd.cut(df["total_engagement"], 
                                   bins=[0, 5, 15, 50, 100, float('inf')],
                                   labels=["Low (0-5)", "Medium (6-15)", "High (16-50)", "Very High (51-100)", "Viral (100+)"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Engagement distribution
        bucket_counts = df["engagement_bucket"].value_counts()
        fig = px.pie(values=bucket_counts.values, names=bucket_counts.index,
                     title="Tweet Engagement Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top performers table
        st.write("**Top 10 Performing Tweets**")
        top_tweets = df.nlargest(10, "total_engagement")[["datetime", "content", "like_count", "retweet_count", "quote_count", "total_engagement", "username"]]
        top_tweets["content"] = top_tweets["content"].str[:100] + "..."
        st.dataframe(top_tweets, use_container_width=True)

def show_account_comparison():
    st.subheader("👥 Account Performance Comparison")
    
    # Account performance metrics
    account_metrics = df.groupby("username").agg({
        "like_count": ["mean", "median", "sum"],
        "retweet_count": ["mean", "median", "sum"],
        "quote_count": ["mean", "median", "sum"],
        "engagement_rate": ["mean", "median"],
        "followers_count": "first",
        "uri": "count"
    }).round(2)
    
    # Flatten column names
    account_metrics.columns = ['_'.join(col).strip() for col in account_metrics.columns]
    account_metrics = account_metrics.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average engagement comparison
        fig = px.bar(
            account_metrics,
            x="username",
            y=["like_count_mean", "retweet_count_mean", "quote_count_mean"],
            title="Average Engagement per Tweet by Account",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement rate comparison
        fig = px.bar(
            account_metrics,
            x="username",
            y="engagement_rate_mean",
            title="Average Engagement Rate by Account"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.write("**Detailed Account Metrics**")
    st.dataframe(account_metrics, use_container_width=True)

def show_content_analysis():
    st.subheader("📝 Content Analysis")
    
    # Tweet length analysis
    df["content_length"] = df["content"].fillna("").str.len()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Length vs engagement scatter
        fig = px.scatter(
            df.sample(min(500, len(df))),  # Sample for performance
            x="content_length",
            y="total_engagement",
            color="username",
            title="Tweet Length vs Engagement",
            labels={"content_length": "Character Count", "total_engagement": "Total Engagement"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Length distribution by account
        fig = px.histogram(
            df,
            x="content_length",
            color="username",
            title="Tweet Length Distribution by Account",
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)

def show_hashtag_and_mentions_analysis():
    st.subheader("🏷️ Hashtag and Mention Analysis")
    
    # Extract hashtags and mentions
    df["hashtag_count"] = df["content"].fillna("").str.count(r"#\w+")
    df["mention_count"] = df["content"].fillna("").str.count(r"@\w+")
    df["url_count"] = df["content"].fillna("").str.count(r"https?://\S+")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Hashtag usage vs engagement
        hashtag_engagement = df.groupby("hashtag_count")["total_engagement"].mean()
        fig = px.bar(
            x=hashtag_engagement.index,
            y=hashtag_engagement.values,
            title="Avg Engagement by Hashtag Count",
            labels={"x": "Number of Hashtags", "y": "Average Engagement"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Mention usage vs engagement
        mention_engagement = df.groupby("mention_count")["total_engagement"].mean()
        fig = px.bar(
            x=mention_engagement.index,
            y=mention_engagement.values,
            title="Avg Engagement by Mention Count",
            labels={"x": "Number of Mentions", "y": "Average Engagement"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # URL usage vs engagement
        url_engagement = df.groupby("url_count")["total_engagement"].mean()
        fig = px.bar(
            x=url_engagement.index,
            y=url_engagement.values,
            title="Avg Engagement by URL Count",
            labels={"x": "Number of URLs", "y": "Average Engagement"}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_performance_insights():
    st.subheader("💡 Performance Insights & Recommendations")
    
    # Calculate key insights
    best_hour = df.groupby("hour")["total_engagement"].mean().idxmax()
    best_day = df.groupby("day_of_week")["total_engagement"].mean().idxmax()
    best_type = df.groupby("tweet_type")["total_engagement"].mean().idxmax()
    
    optimal_length = df.groupby(pd.cut(df["content_length"], bins=5))["total_engagement"].mean().idxmax()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🎯 Optimization Opportunities**")
        st.write(f"• **Best posting time:** {best_hour}:00")
        st.write(f"• **Best posting day:** {best_day}")
        st.write(f"• **Best tweet type:** {best_type}")
        st.write(f"• **Optimal content length:** {optimal_length}")
        
        # Account-specific insights
        for account in df["username"].unique():
            acc_data = df[df["username"] == account]
            avg_engagement = acc_data["total_engagement"].mean()
            st.write(f"• **{account} avg engagement:** {avg_engagement:.1f}")
    
    with col2:
        st.write("**📈 Growth Opportunities**")
        
        # Find underperforming periods
        hourly_avg = df.groupby("hour")["total_engagement"].mean()
        low_hours = hourly_avg[hourly_avg < hourly_avg.quantile(0.3)].index.tolist()
        
        st.write(f"• **Underperforming hours:** {', '.join(map(str, low_hours))}")
        st.write("• **Consider more original content** (higher engagement than retweets)")
        st.write("• **Experiment with hashtag usage** (2-3 hashtags optimal)")
        
        # Engagement rate improvements
        low_engagement_days = df.groupby("day_of_week")["total_engagement"].mean().nsmallest(2).index.tolist()
        st.write(f"• **Focus on:** {', '.join(low_engagement_days)}")

# Update the main navigation to include enhanced analysis
def show_enhanced_account_analysis(account):
    st.header(f"🔍 Enhanced Analysis for @{account}")
    acc_df = df[df["username"] == account].copy()
    acc_df["total_engagement"] = acc_df["like_count"] + acc_df["retweet_count"] + acc_df["quote_count"]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweets", len(acc_df))
    col2.metric("Avg Engagement", f"{acc_df['total_engagement'].mean():.1f}")
    col3.metric("Best Tweet", f"{acc_df['total_engagement'].max()}")
    col4.metric("Engagement Rate", f"{acc_df['engagement_rate'].mean()*100:.2f}%")
    
    # Time-based performance
    st.subheader("📅 Posting Schedule Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly heatmap
        hourly_data = acc_df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_data = hourly_data.reindex(day_order)
        
        fig = px.imshow(
            hourly_data.values,
            x=hourly_data.columns,
            y=hourly_data.index,
            title="Posting Frequency Heatmap",
            labels={"x": "Hour", "y": "Day", "color": "Tweet Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance by time
        time_performance = acc_df.groupby("hour")["total_engagement"].mean()
        fig = px.line(
            x=time_performance.index,
            y=time_performance.values,
            title="Average Engagement by Hour",
            labels={"x": "Hour", "y": "Average Engagement"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Content performance breakdown
    st.subheader("📊 Content Performance Breakdown")
    
    # Tweet type performance for this account
    type_perf = acc_df.groupby("tweet_type").agg({
        "total_engagement": ["mean", "count"],
        "engagement_rate": "mean"
    }).round(2)
    
    type_perf.columns = ['_'.join(col).strip() for col in type_perf.columns]
    st.table(type_perf)
    
    # Recent performance trend
    st.subheader("📈 Recent Performance Trend")
    recent_trend = acc_df.set_index("date")["total_engagement"].rolling(window=7).mean()
    
    fig = px.line(
        x=recent_trend.index,
        y=recent_trend.values,
        title="7-Day Rolling Average Engagement",
        labels={"x": "Date", "y": "7-Day Avg Engagement"}
    )
    st.plotly_chart(fig, use_container_width=True)

# Update the main page logic
if page == "Overview ":
    # Add tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Basic Overview", "Enhanced Analysis", "Temporal Patterns", "Content Analysis", "Insights"])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_enhanced_overview()
        show_account_comparison()
    
    with tab3:
        show_temporal_patterns()
        show_content_performance_analysis()
    
    with tab4:
        show_engagement_distribution_analysis()
        show_content_analysis()
        show_hashtag_and_mentions_analysis()
    
    with tab5:
        show_performance_insights()

elif page == "thedkingdao Analysis":
    # Add tabs for detailed account analysis
    tab1, tab2 = st.tabs(["Basic Analysis", "Enhanced Analysis"])
    
    with tab1:
        show_account_analysis("thedkingdao")
    
    with tab2:
        show_enhanced_account_analysis("thedkingdao")

elif page == "webuildscore Analysis":
    # Add tabs for detailed account analysis
    tab1, tab2 = st.tabs(["Basic Analysis", "Enhanced Analysis"])
    
    with tab1:
        show_account_analysis("webuildscore")
    
    with tab2:
        show_enhanced_account_analysis("webuildscore")
