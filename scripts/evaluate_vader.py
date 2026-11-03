import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

gold = pd.read_csv("data/processed/gold_to_label.csv", encoding="utf-8-sig",
                   encoding_errors="replace")
vader = pd.read_csv("data/processed/comments_vader.csv")[["id", "vader_label", "vader_score"]]

gold["human_label"] = gold["human_label"].astype(str).str.strip().str.lower()
gold = gold[gold["human_label"].isin(["positive", "negative", "neutral"])]
print("labeled rows:", len(gold))

df = gold.merge(vader, on="id", how="left")
labels = ["positive", "neutral", "negative"]

print("\nhuman label distribution:")
print(df["human_label"].value_counts(normalize=True).round(3))

print("\nVADER accuracy:", round(accuracy_score(df["human_label"], df["vader_label"]), 3))
print("\nconfusion matrix (rows = human, columns = VADER):")
print(pd.DataFrame(confusion_matrix(df["human_label"], df["vader_label"], labels=labels),
                   index=labels, columns=labels))
print("\n", classification_report(df["human_label"], df["vader_label"], labels=labels))

# comments where VADER is confidently wrong, for the report
wrong = df[df["human_label"] != df["vader_label"]].copy()
wrong["abs_score"] = wrong["vader_score"].abs()
print("\nbiggest VADER mistakes:")
print(wrong.sort_values("abs_score", ascending=False).head(15)[
    ["clean", "human_label", "vader_label", "vader_score"]].to_string())