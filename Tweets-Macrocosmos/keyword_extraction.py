import requests
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
import logging
from logging.handlers import RotatingFileHandler
import json
from collections import defaultdict
from utils import *

def fetch_keyword_posts_rest(api_key, api_url, keywords, days_back=1, limit=1000):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime('%Y-%m-%dT00:00:00Z')
    end_str = end_date.strftime('%Y-%m-%dT00:00:00Z')

    url = f"{api_url}/api/v1/on_demand_data_request"
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key
    }
    data = {
        "source": "x",
        "keywords": keywords,
        "start_date": start_str,
        "end_date": end_str,
        "limit": limit
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json().get('data', [])

def save_to_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Usage in main:
if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("MAINNET_API_KEY")
    api_url = os.getenv("MAINNET_API_URL")
    APP_NAME = "team_keyword_extraction"
    KEYWORDS = ["Wrexham", "Wrexham AFC", "Ryan Reynolds", "Rob McElhenney", "Hollywood owners"]
    DAYS = 1
    OUTPUT_FILE = "output/team_posts.json"

    results = fetch_keyword_posts_rest(API_KEY, api_url, KEYWORDS, days_back=DAYS, limit=500)
    save_to_json(results, OUTPUT_FILE)
    print(f"Saved {len(results) if isinstance(results, list) else 'some'} records to {OUTPUT_FILE}")