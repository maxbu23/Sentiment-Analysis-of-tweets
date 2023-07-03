import pandas as pd
import re
import nltk
import shap
import json

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from xgboost import XGBClassifier

nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    # Usunięcie linków
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Usunięcie znaków specjalnych i cyfr
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    # Zamiana na małe litery
    text = text.lower()
    # Lematyzacja
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in stop_words]
    text = ' '.join(text)
    return text

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def to_sentiment_category(score):
    if score < 0:
        return 'negative'
    elif score > 0:
        return 'positive'
    else:
        return 'neutral'


df = pd.read_csv('data_sets/trumptweets.csv')
df['processed_text'] = df['content'].apply(preprocess_text)
df['sentiment'] = df['processed_text'].apply(get_sentiment)
df['sentiment_category'] = df['sentiment'].apply(to_sentiment_category)

print("App started...")

X = df['processed_text']
y = df['sentiment_category']

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Wektoryzacja TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Trenowanie modelu SVM
clf = svm.SVC(probability=True)
clf.fit(X_train_vectorized, y_train)

params = clf.get_params()
with open('model.txt','w') as output:
    output.write(json.dumps(params))

# Ocena modelu
accuracy = clf.score(X_test_vectorized, y_test)
print("Accuracy: ", accuracy)


# wyjaśnienie modelu SVM
explainer = shap.KernelExplainer(clf.predict_proba, X_train_vectorized[:50])
print('explainer created')

shap_values = explainer.shap_values(X_train_vectorized[:50])

print('vales shapped')
# wizualizacja
shap.summary_plot(shap_values, X_test_vectorized, 
                  feature_names=vectorizer.get_feature_names_out(), 
                  class_names=['negative','neutral','positive'])