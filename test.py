import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory Usage Before: {process.memory_info().rss / 1024 ** 2} MB")

from joblib import load

# Load the model and vectorizer
model = load('voting_classifier_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

print(f"Memory Usage After: {process.memory_info().rss / 1024 ** 2} MB")

print(97.6-14.8)