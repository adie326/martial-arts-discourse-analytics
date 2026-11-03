import json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

# YouTube comments
for path in RAW.glob("youtube_*.json"):
    for c in json.loads(path.read_text(encoding="utf-8")):
        rows.append({
            "id": f"yt_{c['id']}",
            "source": "youtube",
            "style": c["style"],
            "text": c["text"],
            "score": c["likes"],
            "created_at": c["published_at"],
            "context": c["video_title"],
        })

# Reddit posts (will be used once your API access works)
for path in RAW.glob("reddit_*.json"):
    for p in json.loads(path.read_text(encoding="utf-8")):
        rows.append({
            "id": f"rd_{p['id']}",
            "source": "reddit",
            "style": None,   # fill in from subreddit mapping or keywords
            "text": f"{p.get('title', '')}. {p.get('selftext', '')}".strip(),
            "score": p.get("score"),
            "created_at": pd.to_datetime(p.get("created_utc"), unit="s", utc=True),
            "context": p.get("subreddit"),
        })

df = pd.DataFrame(rows).drop_duplicates(subset="id")
df.to_csv(OUT / "comments_all.csv", index=False)
print(df.groupby(["source", "style"]).size())