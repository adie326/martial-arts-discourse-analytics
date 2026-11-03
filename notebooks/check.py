import json
from pathlib import Path
import pandas as pd

frames = []
for path in Path("data/raw").glob("youtube_*.json"):
    frames.append(pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))))
df = pd.concat(frames, ignore_index=True)

print(df.shape)
print(df["style"].value_counts())            # comments per style
print(df["text"].str.len().describe())       # comment length
print(df["id"].duplicated().sum(), "duplicate ids")
print(df.sample(10, random_state=1)[["style", "text"]].to_string())