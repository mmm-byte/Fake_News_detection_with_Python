!pip install pandas scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Mount Google Drive (if you need to access files from Drive)
from google.colab import drive
drive.mount('/content/drive')

# Provide the path to your news.csv file in Google Drive
csv_path = '/content/drive/MyDrive/news.csv'

# Read the data from the CSV file
df = pd.read_csv(csv_path)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the train set, transform the test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

# Build the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Confusion Matrix:")
print(cm)
