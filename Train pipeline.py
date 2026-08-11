import pandas as pd
import numpy as np
import re, string, json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv('SMSSpamCollection.txt', sep='\t', names=['label','text'])
df = df.drop_duplicates()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['message_length'] = df['clean_text'].apply(len)
df['label_num'] = df['label'].map({'ham':0,'spam':1})

X = df['clean_text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': LinearSVC()
}

results = {}
trained = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }
    trained[name] = model

best_name = max(results, key=lambda k: results[k]['accuracy'])

# Deploy Logistic Regression specifically (matches original model type & gives probability scores for the dashboard UI)
deploy_name = 'Logistic Regression'
best_model = trained[deploy_name]
y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best).tolist()

print("Best model (highest accuracy):", best_name)
print("Deployed model:", deploy_name)
print(results)
print("Confusion matrix:", cm)
print("n_features:", tfidf.get_feature_names_out().shape)

# Save artifacts
joblib.dump(best_model, 'spam_model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

meta = {
    'best_model_name': best_name,
    'deployed_model_name': deploy_name,
    'results': results,
    'confusion_matrix': cm,
    'labels': ['Ham','Spam'],
    'n_train': len(X_train),
    'n_test': len(X_test),
    'class_counts': df['label'].value_counts().to_dict(),
    'msg_len_stats': df.groupby('label')['message_length'].describe().to_dict(),
    'dataset_shape': list(df.shape),
}
with open('metadata.json','w') as f:
    json.dump(meta, f, indent=2, default=str)

# Save a sample of cleaned df for EDA in the dashboard (drop raw could keep small)
df[['label','text','clean_text','message_length']].to_csv('spam_dataset_clean.csv', index=False)
print("Saved artifacts.")