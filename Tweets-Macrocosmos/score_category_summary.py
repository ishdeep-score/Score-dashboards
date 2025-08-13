import pandas as pd
import re

# Load the SCORE_X data
score_df = pd.read_csv("SCORE_X.csv")

# Define your custom categories and classification logic
def classify_custom_category(text):
    text = str(text).lower().strip()
    # Identify articles: only a link or contains article-like keywords
    link_only_pattern = r"^https?://\S+$"
    article_keywords = [
        "read more", "article", "blog", "medium.com", "full story", "breakdown", "deep dive", "thread", "winning metrics"
    ]
    if re.match(link_only_pattern, text):
        return "Article"
    if any(kw in text for kw in ["shot of the week", "play of the week", "miss of the week"]):
        return "Shot/Play/Miss of the Week"
    elif any(kw in text for kw in ["podcast", "clip"]):
        return "Podcast Clips"
    elif any(kw in text for kw in ["introduction", "introducing", "introduced"]):
        return "Introduction Post"
    elif any(kw in text for kw in ["guess the player", "who do you think it is","who should we scout","who's your pick"]):
        return "Guess The Player / Community Scout"
    elif any(kw in text for kw in ["vs", "matchup", "semi-final", "final", "quarter-final", "match", "battle", "face", "meet", "takes on", "takes the trophy", "who wins", "who claims", "who gets the points", "who lifts the trophy"]):
        return "Matchups"
    elif any(kw in text for kw in ["subnet", "sn44", "validator", "bittensor", "ai", "machine learning", "model"]):
        return "Technical/AI/Subnet"
    elif any(kw in text for kw in ["rebrand", "website", "identity", "brand", "design", "logo","roundup", "weekly", "recap", "update", "announcement"]):
        return "Branding/Marketing/Updates"
    else:
        return "Other Football Posts"

# Apply the classification
score_df["Custom_Category"] = score_df["Post text"].fillna("").apply(classify_custom_category)

# Calculate total engagement (adjust columns as needed)
score_df["Total_Engagement"] = (
    score_df["Likes"].fillna(0) +
    score_df["Reposts"].fillna(0) +
    score_df["Replies"].fillna(0) +
    score_df["Bookmarks"].fillna(0) +
    score_df["Profile visits"].fillna(0)
)

# Aggregate average engagement metrics and total posts by custom category
summary = score_df.groupby("Custom_Category").agg(
    Avg_Likes=("Likes", "mean"),
    Avg_Reposts=("Reposts", "mean"),
    Avg_Replies=("Replies", "mean"),
    Avg_Bookmarks=("Bookmarks", "mean"),
    Avg_Profile_Visits=("Profile visits", "mean"),
    Avg_Total_Engagement=("Total_Engagement", "mean"),
    Num_Posts=("Post text", "count")
).reset_index()

# Round averages to 2 decimal points
for col in [
    "Avg_Likes", "Avg_Reposts", "Avg_Replies",
    "Avg_Bookmarks", "Avg_Profile_Visits", "Avg_Total_Engagement"
]:
    summary[col] = summary[col].round(2)

# Save to CSV
summary.to_csv("output/score_category_summary.csv", index=False)
print("Saved summary to output/score_category_summary.csv")