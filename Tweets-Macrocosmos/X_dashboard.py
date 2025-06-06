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
