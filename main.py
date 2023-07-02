import pandas as pd
import re
import nltk
import shap

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

print("App started")

X = df['processed_text']
y = df['sentiment_category']

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Wektoryzacja TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Trenowanie modelu SVM
clf = svm.SVC(probability=True)
clf.fit(X_train_tfidf, y_train)

# Ocena modelu
accuracy = clf.score(X_test_tfidf, y_test)
print("Accuracy: ", accuracy)


# wyjaśnienie modelu SVM
explainer = shap.KernelExplainer(clf.predict_proba, X_train_tfidf)
shap_values = explainer.shap_values(X_test_tfidf)

# wizualizacja
shap.summary_plot(shap_values, X_test_tfidf, feature_names=vectorizer.get_feature_names())

# from collections import defaultdict

# # Inicjalizacja słownika, który przechowuje liczbę wystąpień każdego słowa w kontekście każdej kategorii sentymentu
# word_counts = defaultdict(lambda: defaultdict(int))

# # Iteracja przez każdy wiersz w ramce danych
# for index, row in df.iterrows():
#     # Rozdzielenie przetworzonego tekstu na słowa
#     words = row['processed_text'].split()
#     # Zliczanie wystąpień każdego słowa w kontekście kategorii sentymentu
#     for word in words:
#         word_counts[word][row['sentiment_category']] += 1

# # Wyświetlenie 10 pierwszych wpisów w słowniku
# for i, (word, sentiment_counts) in enumerate(word_counts.items()):
#     print(f"Word: {word}, Sentiment Counts: {dict(sentiment_counts)}")
#     if i == 9:
#         break

# # Tworzenie DataFrame z word_counts
# word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index')

# # Wypełnianie brakujących wartości zerami
# word_counts_df = word_counts_df.fillna(0)

# # Wyświetlanie pierwszych 10 wierszy
# print(word_counts_df.head(10))

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Wybór 20 najczęściej występujących słów
# top_words = word_counts_df.sum(axis=1).nlargest(20).index

# # Tworzenie heatmapy
# plt.figure(figsize=(10, 10))
# sns.heatmap(word_counts_df.loc[top_words], annot=True, fmt=".0f", cmap='YlGnBu')

# plt.title('Liczba wystąpień słów w kontekście sentymentu')
# plt.xlabel('Kategoria sentymentu')
# plt.ylabel('Słowo')

# plt.show()