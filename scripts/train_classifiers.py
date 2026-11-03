import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ---- load both labeled batches ----
b1 = pd.read_csv("data/processed/gold_to_label.csv", encoding="utf-8-sig",
                 encoding_errors="replace")[["id", "clean", "human_label"]]
b2 = pd.read_excel("data/processed/gold_batch2_to_label.xlsx")[["id", "clean", "human_label"]]
df = pd.concat([b1, b2], ignore_index=True).drop_duplicates("id")

df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
df = df[df["human_label"].isin(["positive", "neutral", "negative"])].reset_index(drop=True)
print("labeled rows:", len(df))
print(df["human_label"].value_counts(normalize=True).round(3), "\n")

X, y = df["clean"], df["human_label"]
labels = ["positive", "neutral", "negative"]

# keep emojis and punctuation as tokens (they carry sentiment in this data)
def tfidf():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b|[^\w\s]")

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes":       MultinomialNB(alpha=0.5),
    "Linear SVM":        LinearSVC(C=1.0, class_weight="balanced"),
    "Decision Tree":     DecisionTreeClassifier(max_depth=15, min_samples_leaf=2,
                                                class_weight="balanced", random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary = []

for name, clf in models.items():
    pipe = make_pipeline(tfidf(), clf)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
    summary.append({
        "model": name,
        "accuracy": round(scores["test_accuracy"].mean(), 3),
        "acc_std": round(scores["test_accuracy"].std(), 3),
        "macro_f1": round(scores["test_f1_macro"].mean(), 3),
    })

    pred = cross_val_predict(pipe, X, y, cv=cv)
    print("=" * 60)
    print(name)
    print(pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                       index=labels, columns=labels))
    print(classification_report(y, pred, labels=labels, zero_division=0))

print("=" * 60)
print(pd.DataFrame(summary).to_string(index=False))