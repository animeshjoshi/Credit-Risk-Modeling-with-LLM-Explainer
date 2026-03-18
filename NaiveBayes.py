import json

with open("query_training_data.json", "r") as f:
    data = json.load(f)

# Extract queries and labels
queries = [d["query"] for d in data]
labels = [d["label"] for d in data]

print("Number of queries:", len(queries))
print("Labels:", set(labels))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",   # optional
    ngram_range=(1,2)       # use unigrams + bigrams
)

X = vectorizer.fit_transform(queries)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X, labels)

print("Training complete!")

from sklearn.metrics import classification_report, confusion_matrix

preds = clf.predict(X)
print(classification_report(preds, labels))
print(confusion_matrix(preds, labels))

import joblib

joblib.dump(clf, "query_classifier.pkl")
joblib.dump(vectorizer, "query_vectorizer.pkl")