# evaluate_model.py
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def evaluate_embeddings(embeddings, labels):
    sil_score = silhouette_score(embeddings, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    results = {}

    for model_name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Random Forest", RandomForestClassifier()),
        ("Naive Bayes", GaussianNB()),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[model_name] = acc

    return sil_score, results
