import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# CSV dosyasını yükleyin
df = pd.read_csv("TurkishSMSCollection.csv",sep=";")

# Öznitelikleri ve etiketleri ayırın
X = df['Message']
y = df['GroupText']

# Metin özniteliklerini vektörize edin
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Eğitim ve test setlerini oluşturun
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Lojistik Regresyon modelini oluşturun ve eğitin
model = LogisticRegression()
model.fit(X_train, y_train)

# Yeni cümleler için tahmin olasılıklarını alın
def predict_proba(sentence):
    sentence_vec = vectorizer.transform([sentence])
    probabilities = model.predict_proba(sentence_vec)[0]
    return probabilities

# Test
# API ile buraya belirtilen cümle girilecek
test_sentence = ""
probabilities = predict_proba(test_sentence)
print("Normal Grup Olasılığı:", round(probabilities[0],2))
print("Dolandırıcı Grup Olasılığı:", round(probabilities[1],2))
