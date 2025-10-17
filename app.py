import nltk
import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load dataset
news_df = pd.read_csv('train.csv')
news_df.fillna(' ', inplace=True)
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Stemming function
ps = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower()
    content = content.split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return " ".join(content)

# Apply stemming
news_df['content'] = news_df['content'].apply(stemming)

# Prepare features and labels
X = news_df['content'].values
y = news_df['label'].values 

# Vectorize text
vector = TfidfVectorizer()
X = vector.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit Web App
st.title('Fake News Detector')
input_text = st.text_input('Enter News Article')

# Prediction function
def prediction(input_text):
    input_text = stemming(input_text)  # Apply preprocessing
    input_data = vector.transform([input_text])
    pred = model.predict(input_data)
    return pred[0]

# Display result
if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else:
        st.write('The News is Real')
