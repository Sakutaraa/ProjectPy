import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('spam.csv', encoding='latin-1')

# Preprocess the data
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Convert the label to a numerical representation
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into features and labels
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Transform the text data into numerical features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create the model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

def predict_message(message):
  # Preprocess the message
  message = [message]
  message = vectorizer.transform(message)

  # Make a prediction
  prediction = model.predict(message)[0]
  probability = model.predict_proba(message)[0][prediction]

  if prediction == 0:
    label = "ham"
  else:
    label = "spam"

  return [probability, label]
