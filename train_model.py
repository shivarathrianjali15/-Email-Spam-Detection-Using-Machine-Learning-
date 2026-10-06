import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("spam.csv")

# Input and output
X = data["message"]
y = data["label"]

# Create Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "spam_model.pkl")

print("Model trained successfully!")