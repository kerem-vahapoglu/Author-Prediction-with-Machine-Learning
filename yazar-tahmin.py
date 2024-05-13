import numpy as np
import pandas as pd
import os
import re
import re
from snowballstemmer import TurkishStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score


dosyalar = os.listdir("makine_ögrenimi\\yazilar")

df = pd.DataFrame()
mylist = []

print(dosyalar)

for dosya in dosyalar:
    text = None
    ayirma = "*******************"
    temp = {}
    with open(f"makine_ögrenimi\\yazilar\\{dosya}", "r", encoding='utf-8') as file:
        text = file.read().strip()
        text = re.sub(r"(\n){2,}", ayirma, text).replace("\n"," ")
        temp['Yazar'] = dosya.replace(".txt","")
        temp['Yazi'] = text.split(ayirma)
    mylist.append(temp)


for i in range(len((dosyalar))):
    df = pd.concat([df,pd.DataFrame(mylist[i])])    


turkish_stopwords = None

with open("makine_ögrenimi\\stop-words.txt", "r", encoding='utf-8') as file:
    turkish_stopwords = set(file.read().replace("\n"," ").split())

print(turkish_stopwords)


def veri_temizligi(text):
    metin = re.sub("[^a-zA-ZçÇğĞıİöÖşŞüÜ]", " ", text).lower()
    kelimeler = metin.split()
    kelimeler = [i for i in kelimeler if not i in turkish_stopwords or len(i) == 1]
    
    return kelimeler

def kelime_kokunu_bul_turkish_stemmer(kelime):
    stemmer = TurkishStemmer()
    return stemmer.stemWord(kelime)

for i in range(len(df['Yazi'])):

    kokler = list(map(kelime_kokunu_bul_turkish_stemmer, veri_temizligi(df.iloc[i,1])))
    metin_son = " ".join(kokler)
    df.iloc[i,1] = metin_son


cv = CountVectorizer()

x = cv.fit_transform(df['Yazi'].tolist()).toarray()
y = df['Yazar'].values


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


model = LogisticRegression(C=0.000783, random_state=42)
model.fit(X_train,y_train)


tahminler = model.predict(X_test)



cm = confusion_matrix(y_test, tahminler)
AccScore = accuracy_score(y_test, tahminler)

print(f"Confisuon Matrix : \n {cm}\n")
print(f"Accuracy Score => {AccScore}")


denemedf = pd.DataFrame(y_train)
denemedf.value_counts()


print(f"Test Score = {model.score(X_test,y_test)}")
print(f"Train Score = {model.score(X_train,y_train)}")
