import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("data/processed/comments_focus.csv")
sia = SentimentIntensityAnalyzer()

df["vader_score"] = df["clean"].apply(lambda t: sia.polarity_scores(t)["compound"])

def to_label(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"

df["vader_label"] = df["vader_score"].apply(to_label)
print(df["vader_label"].value_counts(normalize=True).round(3))
df.to_csv("data/processed/comments_vader.csv", index=False)

# 300 random comments for you to label by hand (VADER's label is hidden on purpose)
sample = df.sample(300, random_state=42)[["id", "clean"]]
sample["human_label"] = ""
sample.to_csv("data/processed/gold_to_label.csv", index=False, encoding="utf-8-sig")