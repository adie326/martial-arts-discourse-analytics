import json
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://www.reddit.com"
HEADERS = {"User-Agent": "martial-arts-research/0.1 (college project)"}
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# subreddit -> style label
SUBREDDITS = {
    "bjj": "BJJ",
    "karate": "Karate",
    "kravmaga": "Krav Maga",
    "taekwondo": "Taekwondo",
    "Boxing": "Boxing",
    "MMA": "MMA",
    "martialarts": "General",   # style gets tagged later via keywords
}

KEEP = ["id", "subreddit", "title", "selftext", "score", "upvote_ratio",
        "num_comments", "created_utc", "permalink", "link_flair_text"]


def fetch_subreddit(subreddit, sort="top", time_filter="year", target=1000):
    url = f"{BASE_URL}/r/{subreddit}/{sort}.json"
    posts, after = [], None

    while len(posts) < target:
        params = {"limit": 100}
        if sort in ("top", "controversial"):
            params["t"] = time_filter
        if after:
            params["after"] = after

        response = httpx.get(url, params=params, headers=HEADERS, timeout=30)
        print(f"fetching {response.url}")

        if response.status_code == 429:      # rate limited, wait and retry
            time.sleep(30)
            continue
        response.raise_for_status()

        data = response.json()["data"]
        posts.extend(child["data"] for child in data["children"])
        after = data["after"]
        if not after:                        # no more pages
            break
        time.sleep(2)

    return posts[:target]


def save_raw(posts, subreddit, sort="top"):
    stem = f"reddit_{subreddit.lower()}_{sort}_{date.today().isoformat()}"
    path = RAW_DIR / f"{stem}.json"
    path.write_text(json.dumps(posts, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(posts)} posts -> {path}")


if __name__ == "__main__":
    save_raw(fetch_subreddit("bjj"), "bjj")