import json
import os
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# style -> search queries (opinion-style queries give the most useful comments)
QUERIES = {
    # "BJJ": ["is BJJ worth it", "BJJ for self defense", "BJJ vs boxing"],
    "Karate": ["is karate useful", "karate for self defense", "traditional karate vs MMA"],
    "Krav Maga": ["is krav maga effective", "krav maga vs BJJ"],
    "Taekwondo": ["is taekwondo effective", "taekwondo vs karate"],
    "Boxing": ["boxing for self defense", "boxing vs MMA"],
    "MMA": ["best martial art for MMA", "MMA vs traditional martial arts"],
}


def search_videos(query, max_videos=10):
    response = youtube.search().list(
        q=query, part="snippet", type="video",
        maxResults=max_videos, order="relevance", relevanceLanguage="en",
    ).execute()
    return [(i["id"]["videoId"], i["snippet"]["title"]) for i in response["items"]]


def fetch_comments(video_id, max_comments=300):
    comments, token = [], None
    while len(comments) < max_comments:
        try:
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=token, textFormat="plainText", order="relevance",
            ).execute()
        except HttpError as e:
            print(f"  skipping {video_id}: {e.resp.status} (comments likely disabled)")
            break

        for item in response["items"]:
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "id": item["id"],
                "video_id": video_id,
                "text": s["textDisplay"],
                "likes": s["likeCount"],
                "published_at": s["publishedAt"],
            })

        token = response.get("nextPageToken")
        if not token:
            break
        time.sleep(0.5)
    return comments[:max_comments]


def collect_style(style, queries):
    all_comments, seen_videos = [], set()
    for query in queries:
        for video_id, title in search_videos(query):
            if video_id in seen_videos:
                continue
            seen_videos.add(video_id)
            print(f"[{style}] {title[:60]}")
            for c in fetch_comments(video_id):
                c.update({"style": style, "video_title": title, "query": query})
                all_comments.append(c)
    return all_comments


def save_raw(comments, style):
    stem = f"youtube_{style.lower().replace(' ', '_')}_{date.today().isoformat()}"
    path = RAW_DIR / f"{stem}.json"
    path.write_text(json.dumps(comments, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(comments)} comments -> {path}")


if __name__ == "__main__":
    # test with one style first
    # save_raw(collect_style("BJJ", QUERIES["BJJ"]), "BJJ")

    # once it works, replace the lines above with:
    for style, queries in QUERIES.items():
        save_raw(collect_style(style, queries), style)