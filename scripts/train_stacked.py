import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---- labeled data (both batches) ----
b1 = pd.read_csv("data/processed/gold_to_label.csv", encoding="utf-8-sig",
                 encoding_errors="replace")[["id", "clean", "human_label"]]
b2 = pd.read_excel("data/processed/gold_batch2_to_label.xlsx")[["id", "clean", "human_label"]]
df = pd.concat([b1, b2], ignore_index=True).drop_duplicates("id")
df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
df = df[df["human_label"].isin(["positive", "neutral", "negative"])].reset_index(drop=True)
print("labeled rows:", len(df))

# ---- pretrained features (RoBERTa class probabilities + VADER score) ----
clf = pipeline("text-classification",
               model="cardiffnlp/twitter-roberta-base-sentiment-latest",
               truncation=True, max_length=256, batch_size=16, top_k=None)
rows = []
for scores in clf(df["clean"].tolist()):
    d = {s["label"].lower(): s["score"] for s in scores}
    rows.append([d["negative"], d["neutral"], d["positive"]])
rob = np.array(rows)

sia = SentimentIntensityAnalyzer()
vader = np.array([sia.polarity_scores(t)["compound"] for t in df["clean"]]).reshape(-1, 1)
dense = np.hstack([rob, vader])

pd.DataFrame(dense, columns=["p_neg", "p_neu", "p_pos", "vader"]).assign(id=df["id"]).to_csv(
    "data/processed/labeled_features.csv", index=False)

# ---- raw pretrained tools on the same rows (fair comparison) ----
labels = ["positive", "neutral", "negative"]
y = df["human_label"]
rob_label = np.array(["negative", "neutral", "positive"])[rob.argmax(axis=1)]
vader_label = np.where(vader[:, 0] >= 0.05, "positive",
                       np.where(vader[:, 0] <= -0.05, "negative", "neutral"))
print("\nraw RoBERTa accuracy:", round(accuracy_score(y, rob_label), 3),
      "| macro F1:", round(f1_score(y, rob_label, average="macro"), 3))
print("raw VADER accuracy:  ", round(accuracy_score(y, vader_label), 3),
      "| macro F1:", round(f1_score(y, vader_label, average="macro"), 3))

# ---- cross-validated comparison ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
text = df["clean"]

def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w\w+\b|[^\w\s]")

def evaluate(name, build):
    accs, f1s = [], []
    oof = np.empty(len(df), dtype=object)
    for tr, te in cv.split(text, y):
        Xtr, Xte = build(tr, te)
        m = LogisticRegression(C=2, class_weight="balanced", max_iter=3000)
        m.fit(Xtr, y.iloc[tr])
        pred = m.predict(Xte)
        oof[te] = pred
        accs.append(accuracy_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred, average="macro"))
    print(f"{name:32s} acc {np.mean(accs):.3f} (+/-{np.std(accs):.3f})  macro-F1 {np.mean(f1s):.3f}")
    return oof

def only_tfidf(tr, te):
    v = tfidf()
    return v.fit_transform(text.iloc[tr]), v.transform(text.iloc[te])

def only_dense(tr, te):
    return dense[tr], dense[te]

def both(tr, te):
    v = tfidf()
    a, b = v.fit_transform(text.iloc[tr]), v.transform(text.iloc[te])
    return sp.hstack([a, sp.csr_matrix(dense[tr] * 2)]).tocsr(), \
           sp.hstack([b, sp.csr_matrix(dense[te] * 2)]).tocsr()

print()
evaluate("TF-IDF only", only_tfidf)
evaluate("RoBERTa + VADER scores only", only_dense)
oof = evaluate("TF-IDF + RoBERTa + VADER", both)

print("\nconfusion matrix, combined model (rows = human):")
print(pd.DataFrame(confusion_matrix(y, oof, labels=labels), index=labels, columns=labels))
print(classification_report(y, oof, labels=labels, zero_division=0))