
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("real_spam.csv")

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label", "message"])

X = df["message"].astype(str)
y = df["label"].astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Three ML algorithms
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

results = {}

print("\n===== SMS-SHIELD MODEL COMPARISON =====")

for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"\n{name} Accuracy: {accuracy * 100:.2f}%")

# Select best model
best_name = max(results, key=results.get)
best_model = models[best_name]

print("\nBest Algorithm:", best_name)

print("\nClassification Report:")
print(classification_report(
    y_test,
    best_model.predict(X_test_tfidf),
    target_names=["NOT SPAM", "SPAM"]
))

# Save best model and vectorizer
joblib.dump(best_model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nBEST MODEL SAVED SUCCESSFULLY!")










