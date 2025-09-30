# train_knn_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv('data.csv')
df.columns = ['description', 'label']
df.dropna(inplace=True)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['description'])
y = df['label']

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)  # You can increase for top-k
knn.fit(X, y)

# Save model and vectorizer
joblib.dump(knn, 'knn_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print("✅ Model and vectorizer saved.")
