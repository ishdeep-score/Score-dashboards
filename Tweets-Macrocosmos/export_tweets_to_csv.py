import sqlite3
import pandas as pd

# Columns to export
COLUMNS = [
    'uri', 'datetime', 'content', 'username', 'followers_count', 'following_count',
    'like_count', 'retweet_count', 'quote_count', 'is_reply', 'is_retweet', 'is_quote',
    'in_reply_to_user_id', 'in_reply_to_username'
]

def export_to_csv(db_path='tweets.db', csv_path='tweets.csv'):
    conn = sqlite3.connect(db_path)
    query = f"SELECT {', '.join(COLUMNS)} FROM tweets"
    df = pd.read_sql_query(query, conn)
    df.to_csv(csv_path, index=False)
    conn.close()
    print(f"Exported {len(df)} rows to {csv_path}")

if __name__ == "__main__":
    export_to_csv() 