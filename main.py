import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

#Wczytanie danych z pliku
data = pd.read_csv('iris.csv')

#Podział danych na macierz cech i wektor etykiet
X = data.drop('Species', axis=1)
y = data['Species']

# 70% treningowy, 30% testowy
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=9)
#Wygenerowanie wykresów
sns.pairplot(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']], hue='Species')
plt.show()

#Inicjalizacja modelu klasyfikatora KNN
knn = KNeighborsClassifier(n_neighbors=2)
#Dopasowanie modelu do danych treningowych
knn.fit(X_train, y_train)
#Wykonanie prognoz na zbiorze testowym
y_pred = knn.predict(X_test)

#Obliczenie dokładności, raportu klasyfikacji i macierzy konfuzji
scores = metrics.accuracy_score(y_test, y_pred)
classification_rep_knn = classification_report(y_test, y_pred)
conf_matrix_knn = confusion_matrix(y_test, y_pred)

#Wyswitlic wyniki obliczen
print("Wyniki KNN:")
print("Dokładność:", scores)
print("Raport:\n", classification_rep_knn)
print("Macierz:\n", conf_matrix_knn)

print()
print()

#Wykonanie prognoz dla trzech zestawów nowych danych
X_new = np.array([[6.3, 2.7, 4.9, 1.9]])
prediction = knn.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

X_new = np.array([[5.9, 2.7, 5.1, 1.6]])
prediction = knn.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

X_new = np.array([[5.2, 4.2, 1.5, 0.1]])
prediction = knn.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

print()
print()

#Wczytanie danych z pliku
data = pd.read_csv('iris.csv')

#Podział danych na macierz cech i wektor etykiet
X = data.drop('Species', axis=1)
y = data['Species']

# 70% treningowy, 30% testowy
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.3, random_state=10)

model = svm.SVC(kernel='linear', C=2)
model.fit(X_train, y_train)
#Ocena modelu na zbiorze testowym
y_pred = model.predict(X_test)

#Obliczenie dokładności, raportu klasyfikacji i macierzy konfuzji
result_svm = metrics.accuracy_score(y_test, y_pred)
classification_rep_svm = classification_report(y_test, y_pred)
conf_matrix_svm = confusion_matrix(y_test, y_pred)

#Wyswitlic wyniki obliczen
print("Wyniki SVM:")
print("Dokładność:", result_svm)
print("Raport:\n", classification_rep_svm)
print("Macierz:\n", conf_matrix_svm)

#Wykonanie prognoz dla trzech zestawów nowych danych
X_new = np.array([[6.3, 2.7, 4.9, 1.9]])
prediction = model.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

X_new = np.array([[5.9, 2.7, 5.1, 1.6]])
prediction = model.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

X_new = np.array([[5.2, 4.2, 1.5, 0.1]])
prediction = model.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

print()
print()

#Inicjalizacja i dopasowanie modelu Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=10)
model_rf.fit(X_train, y_train)

#Ocena modelu na zbiorze testowym
y_pred = model_rf.predict(X_test)

#Obliczenie dokładności, raportu klasyfikacji i macierzy konfuzji
result_RF = metrics.accuracy_score(y_test, y_pred)
classification_rep_fr = classification_report(y_test, y_pred)
conf_matrix_fr = confusion_matrix(y_test, y_pred)

#Wyswitlic wyniki obliczen
print("Wyniki Random Forest:")
print("Dokładność:", result_RF)
print("Raport:\n", classification_rep_fr)
print("Macierz:\n", conf_matrix_fr)

#Wykonanie prognoz dla trzech zestawów nowych danych
X_new = np.array([[6.3, 2.7, 4.9, 1.9]])
prediction = model_rf.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

X_new = np.array([[5.9, 2.7, 5.1, 1.6]])
prediction = model_rf.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

X_new = np.array([[5.2, 4.2, 1.5, 0.1]])
prediction = model_rf.predict(X_new)
print("Jaki kwiat?: {}".format(prediction))

print()
print()

#Obliczanie waznosci
feature_importance = model_rf.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Ważność cech:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# wykres ważności cech
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_feature_importance)), [val[1] for val in sorted_feature_importance], align='center')
plt.yticks(range(len(sorted_feature_importance)), [val[0] for val in sorted_feature_importance])
plt.title('Ważność cech')
plt.show()

print()
#Wykorzystanie walidacji krzyżowej
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=9)

#Wyświetlenie średniej dokładności i odchylenia standardowego
scores = cross_val_score(model_rf, X, y, cv=kf, scoring='accuracy')
mean_accuracy = scores.mean()
std_accuracy = scores.std()

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność: {i + 1}: {accuracy}")

print()

#Grid Search
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [1, 10, 20],
    'min_samples_split': [5, 10, 30],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepsza ocena (dokładność):", grid_search.best_score_)
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)
best_model = grid_search.best_estimator_
