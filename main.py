from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

file_path = "ulasan.csv"
df = pd.read_csv(file_path)

df = df.dropna(subset=['Sentimen'])

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Sentimen'])

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(df['Ulasan'])

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, df['label'])

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

target_names = label_encoder.classes_.tolist()
target_names = [str(label) for label in target_names]

report = classification_report(y_test, y_pred, target_names=['Negatif', 'Netral', 'Positif'])

def prediksi_sentimen(ulasan_baru):
    new_counts = count_vect.transform([ulasan_baru])
    new_tfidf = tfidf_transformer.transform(new_counts)
    prediction = model.predict(new_tfidf)
    return label_encoder.inverse_transform(prediction)[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil_prediksi = None
    if request.method == 'POST':
        ulasan_baru = request.form['ulasan']
        hasil_prediksi_label = prediksi_sentimen(ulasan_baru)
        
        if hasil_prediksi_label == 1:
            hasil_prediksi = "Positif"
        elif hasil_prediksi_label == -1:
            hasil_prediksi = "Negatif"
        else:
            hasil_prediksi = "Netral"
            
    return render_template('index.html', hasil_prediksi=hasil_prediksi)

if __name__ == '__main__':
    app.run(debug=True)
