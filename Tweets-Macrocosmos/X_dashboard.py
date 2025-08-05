import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
        return "Guess The Player / Community Scout"
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
        "Impressions": "sum",
        "Engagement": "sum",
        "Likes": "sum",
        "Reposts": "sum",
        "Replies": "sum",
        "New follows": "sum"
    })
    # Add engagement rate for each category
    category_stats["Engagement Rate %"] = (category_stats["Engagement"] / category_stats["Impressions"] * 100).round(2)
    st.table(category_stats)

    # Post Type Analysis (Articles, Threads, Posts)
    st.subheader("Performance by Post Type (Articles vs Threads vs Posts)")
    type_stats = score_df.groupby("Type").agg({
        "Impressions": "sum",
        "Engagement": "sum",
        "Likes": "sum",
        "Reposts": "sum",
        "Replies": "sum",
        "New follows": "sum"
    })
    # Add engagement rate for each type
    type_stats["Engagement Rate %"] = (type_stats["Engagement"] / type_stats["Impressions"] * 100).round(2)
    st.table(type_stats)
    
    st.bar_chart(type_stats["Engagement"])

    # Non-Football Category Breakdown
    st.subheader("Non-Football Post Categories")
    non_football_posts = score_df[score_df["Category"] == "Other"]
    if len(non_football_posts) > 0:
        st.bar_chart(non_football_posts["Non_Football_Category"].value_counts())
        
        st.subheader("Non-Football Category Performance")
        non_football_stats = non_football_posts.groupby("Non_Football_Category").agg({
            "Impressions": "sum",
            "Engagement": "sum",
            "Likes": "sum",
            "Reposts": "sum",
            "New follows": "sum"
        })
        non_football_stats["Engagement Rate %"] = (non_football_stats["Engagement"] / non_football_stats["Impressions"] * 100).round(2)
        st.table(non_football_stats)

    # Football Category Analysis
    st.subheader("Football Post Categories")
    football_posts = score_df[score_df["Category"] == "Football"]
    if len(football_posts) > 0:
        st.bar_chart(football_posts["Football_Category"].value_counts())
        
        st.subheader("Football Category Performance")
        football_stats = football_posts.groupby("Football_Category").agg({
            "Impressions": "sum",
            "Engagement": "sum",
            "Likes": "sum",
            "Reposts": "sum",
            "New follows": "sum"
        })
        football_stats["Engagement Rate %"] = (football_stats["Engagement"] / football_stats["Impressions"] * 100).round(2)
        st.table(football_stats)

    # Top Performing Posts by Type
    st.subheader("Top Posts by Type")
    for post_type in score_df["Type"].unique():
        type_posts = score_df[score_df["Type"] == post_type].sort_values("Engagement", ascending=False).head(3)
        st.write(f"**Top {post_type}s:**")
        for _, row in type_posts.iterrows():
            st.write(f"- {row['Post text'][:100]}... | Total Engagement: {row['Engagement']} | Impressions: {row['Impressions']:,}")
        st.write("")

    # Simplified Insights
    st.subheader("Key Performance Insights")
    
    # Calculate totals for each type
    thread_total = score_df[score_df["Type"] == "Thread"]["Engagement"].sum()
    post_total = score_df[score_df["Type"] == "Post"]["Engagement"].sum()
    article_total = score_df[score_df["Type"] == "Article"]["Engagement"].sum()
    
    thread_impressions = score_df[score_df["Type"] == "Thread"]["Impressions"].sum()
    post_impressions = score_df[score_df["Type"] == "Post"]["Impressions"].sum()
    article_impressions = score_df[score_df["Type"] == "Article"]["Impressions"].sum()
    
    st.markdown(f"""
    **Post Type Total Performance:**
    - Threads: {thread_total:,} total engagement from {thread_impressions:,} impressions ({(thread_total/thread_impressions*100):.2f}% rate)
    - Regular posts: {post_total:,} total engagement from {post_impressions:,} impressions ({(post_total/post_impressions*100):.2f}% rate)
    - Articles: {article_total:,} total engagement from {article_impressions:,} impressions ({(article_total/article_impressions*100):.2f}% rate)
    
    **Key Insights:**
    - Best performing content type: {max([("Threads", thread_total/thread_impressions), ("Posts", post_total/post_impressions), ("Articles", article_total/article_impressions)], key=lambda x: x[1])[0]}
    - Total new followers gained: {score_df['New follows'].sum():,}
    - Most engaging category: {score_df.groupby('Category')['Engagement'].sum().idxmax()}
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
    col4.metric("Overall Engagement Rate", f"{(dking_df['Engagement'].sum() / dking_df['Impressions'].sum() * 100):.2f}%")

    # DKING Category Analysis
    st.subheader("DKING Post Categories")
    st.bar_chart(dking_df["Category"].value_counts())
    
    st.subheader("Performance by Category")
    dking_stats = dking_df.groupby("Category").agg({
        "Impressions": "sum",
        "Engagement": "sum",
        "Likes": "sum",
        "Reposts": "sum",
        "Replies": "sum",
        "New follows": "sum"
    })
    dking_stats["Engagement Rate %"] = (dking_stats["Engagement"] / dking_stats["Impressions"] * 100).round(2)
    st.table(dking_stats)

    # Top DKING Posts
    st.subheader("Top DKING Posts")
    top_dking = dking_df.sort_values("Engagement", ascending=False).head(5)
    st.table(top_dking[["Date", "Post text", "Category", "Engagement", "Impressions", "Likes"]])

    # Enhanced DKING Content Analysis
    st.subheader("Enhanced DKING Content Analysis")
    analyze_dking_content()
    content_stats = dking_df.groupby("Content_Category").agg({
        "Impressions": "sum",
        "Engagement": "sum",
        "Likes": "sum",
        "Reposts": "sum",
        "Replies": "sum",
        "New follows": "sum"
    })
    content_stats["Engagement Rate %"] = (content_stats["Engagement"] / content_stats["Impressions"] * 100).round(2)
    st.table(content_stats)

    # DKING Insights
    st.subheader("DKING Performance Insights")
    best_category = dking_stats["Engagement"].idxmax()
    best_rate = dking_stats.loc[best_category, "Engagement Rate %"]
    
    st.markdown(f"""
    **Top Performing Category:** {best_category} ({best_rate}% engagement rate)
    **Total Followers Gained:** {dking_df['New follows'].sum():,}
    **Total Profile Visits:** {dking_df['Profile visits'].sum():,}
    **Best Post:** {dking_df.loc[dking_df['Engagement'].idxmax(), 'Post text'][:100]}...
    """)

# ...rest of existing code...
