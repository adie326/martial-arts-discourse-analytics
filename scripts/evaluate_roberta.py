import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline

gold = pd.read_csv("data/processed/gold_to_label.csv", encoding="utf-8-sig",
                   encoding_errors="replace")
gold["human_label"] = gold["human_label"].astype(str).str.strip().str.lower()
gold = gold[gold["human_label"].isin(["positive", "negative", "neutral"])]
vader = pd.read_csv("data/processed/comments_vader.csv")[["id", "vader_label"]]
df = gold.merge(vader, on="id", how="left")

clf = pipeline("text-classification",
               model="cardiffnlp/twitter-roberta-base-sentiment-latest",
               truncation=True, max_length=256, batch_size=16)
preds = clf(df["clean"].tolist())
df["roberta_label"] = [p["label"].lower() for p in preds]
df["roberta_conf"] = [p["score"] for p in preds]

labels = ["positive", "neutral", "negative"]
print("labeled rows:", len(df))
for col in ["vader_label", "roberta_label"]:
    print(col, "accuracy:", round(accuracy_score(df["human_label"], df[col]), 3))

print("\nRoBERTa confusion matrix (rows = human, columns = model):")
print(pd.DataFrame(confusion_matrix(df["human_label"], df["roberta_label"], labels=labels),
                   index=labels, columns=labels))
print("\n", classification_report(df["human_label"], df["roberta_label"], labels=labels))

wrong = df[df["human_label"] != df["roberta_label"]].copy()
wrong["clean"] = wrong["clean"].str[:150]          # keeps the printout readable
print("\nbiggest RoBERTa mistakes:")
print(wrong.sort_values("roberta_conf", ascending=False).head(15)[
    ["clean", "human_label", "roberta_label", "roberta_conf"]].to_string())

df.to_csv("data/processed/gold_with_predictions.csv", index=False)