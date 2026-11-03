import re
import pandas as pd

df = pd.read_csv("data/processed/comments_all.csv")
df = df.rename(columns={"style": "video_style"})
df = df.drop_duplicates("id")

URL = r"https?://\S+|www\.\S+"
TIMESTAMP = r"\b\d{1,2}:\d{2}(?::\d{2})?\b"

df["clean"] = (df["text"].astype(str)
               .str.replace(URL, "", regex=True)
               .str.replace(TIMESTAMP, "", regex=True)
               .str.replace(r"\s+", " ", regex=True)
               .str.strip())

df["n_words"] = df["clean"].str.split().str.len()
df = df[(df["n_words"] >= 5) & (df["n_words"] <= 200)]
print("after length filter:", len(df))

STYLE_PATTERNS = {
    # "BJJ": r"\b(bjj|jiu[- ]?jitsu)\b",
    # "Karate": r"\b(karate|kyokushin|shotokan)\b",
    # "Taekwondo": r"\b(taekwondo|tkd|tae ?kwon ?do)\b",
    # "Krav Maga": r"\bkrav\b",
    # "Boxing": r"\b(boxing|boxer|boxers)\b",
    # "MMA": r"\b(mma|ufc)\b",

    "BJJ": r"\b(?:bjj|jiu[- ]?jitsu)\b",
    "Karate": r"\b(?:karate|kyokushin|shotokan)\b",
    "Taekwondo": r"\b(?:taekwondo|tkd|tae ?kwon ?do)\b",
    "Krav Maga": r"\bkrav\b",
    "Boxing": r"\b(?:boxing|boxer|boxers)\b",
    "MMA": r"\b(?:mma|ufc)\b",
}
for style, pattern in STYLE_PATTERNS.items():
    df[f"m_{style}"] = df["clean"].str.contains(pattern, case=False, regex=True)

mention_cols = [c for c in df.columns if c.startswith("m_")]
df["n_styles"] = df[mention_cols].sum(axis=1)

print("\nstyles mentioned per comment:")
print(df["n_styles"].value_counts().sort_index())
print("\nmentions per style:")
print(df[mention_cols].sum())

focus = df[df["n_styles"] >= 1]
print("\nfocus set size:", len(focus))
focus.to_csv("data/processed/comments_focus.csv", index=False)
df.to_csv("data/processed/comments_clean.csv", index=False)