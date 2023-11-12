#1.	Chargement des librairies Python
from time import *
t_départ=time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2.	Chargement des données
dataset = pd.read_csv("temp.csv")
dataset.head()
X = dataset.drop('Amplitude', axis=1)
y = dataset['Classe']

#3.les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70)

#4.	Entraînement de l'algorithme
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#5.	Évaluation de l'algorithme
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, adjusted_mutual_info_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Types d'algorithme:Classification & Régression ")
print("nombre de paramètre: 2.000")
print("Précision moyenne ",accuracy_score(y_test, y_pred))
print("Taux  d'ajustement",adjusted_mutual_info_score(y_test, y_pred))
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
t_fin=time()
print("tempsd'exécution:",t_fin-t_départ,"secondes")
