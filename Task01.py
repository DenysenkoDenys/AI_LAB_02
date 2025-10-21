import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

input_file = 'income_data.txt'

X_raw = []
y_labels = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

print(f"Спроба завантажити дані з файлу: {input_file}")
try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break

            if '?' in line:
                continue


            data = line.strip().split(', ')

            if not data or len(data) < 15:
                continue

            label = data[-1]

            if label == '<=50K' and count_class1 < max_datapoints:
                X_raw.append(data[:-1])
                y_labels.append(label)
                count_class1 += 1

            elif label == '>50K' and count_class2 < max_datapoints:
                X_raw.append(data[:-1])
                y_labels.append(label)
                count_class2 += 1

except FileNotFoundError:
    print(f"\nПОМИЛКА: Файл '{input_file}' не знайдено.")
    print("Переконайтеся, що файл 'income_data.txt' існує у кореневому каталозі.")
    sys.exit(1)

print(f"Завантажено точок даних:\nКлас <= 50K: {count_class1}\nКлас > 50K: {count_class2}")


X = np.array(X_raw)
y_labels = np.array(y_labels)

# workclass (1), education (3), marital-status (5), occupation (6),
# relationship (7), race (8), sex (9), native-country (13)
categorical_feature_indices = [1, 3, 5, 6, 7, 8, 9, 13]

label_encoders = {}
num_features = X.shape[1]

X_encoded = np.empty(X.shape, dtype=float)


for i in range(num_features):
    if i in categorical_feature_indices:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoders[i] = le
    else:
        X_encoded[:, i] = X[:, i].astype(float)

X = X_encoded.astype(int)
y_encoder = preprocessing.LabelEncoder()
y = y_encoder.fit_transform(y_labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
print("-" * 50)
print("Початок навчання класифікатора на навчальному наборі (80/20)...")
classifier.fit(X_train, y_train)
print("Навчання класифікатора завершено.")

y_test_pred = classifier.predict(X_test)

f1_scores_cv = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
f1_mean_cv = f1_scores_cv.mean()

print("\n--- Метрики Класифікації ---")
print("I. Результати на основі перехресної перевірки (CV=3):")
print(f"Усереднений F1 score: {round(100 * f1_mean_cv, 2)}%")
print("-" * 50)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1_test = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print("II. Метрики на тестовому наборі (20% даних):")
print(f"Accuracy (Акуратність): {round(100 * accuracy, 2)}%")
print(f"Precision (Точність): {round(100 * precision, 2)}%")
print(f"Recall (Повнота): {round(100 * recall, 2)}%")
print(f"F1 Score (F1-міра): {round(100 * f1_test, 2)}%")
print("-" * 50)

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White',
              'Male', '0', '0', '40', 'United-States']

input_data_encoded = np.empty(len(input_data), dtype=int)

for i, item in enumerate(input_data):
    if i in categorical_feature_indices:
        try:
            input_data_encoded[i] = label_encoders[i].transform([item])[0]
        except ValueError:
            input_data_encoded[i] = 0
    else:
        try:
            input_data_encoded[i] = int(item)
        except ValueError:
            input_data_encoded[i] = 0

input_data_reshaped = input_data_encoded.reshape(1, -1)

predicted_class_encoded = classifier.predict(input_data_reshaped)[0]

predicted_label = y_encoder.inverse_transform([predicted_class_encoded])[0]

print(f"Тестова точка даних (ознаки): {input_data}")
print(f"Прогнозований клас (дохід): {predicted_label}")
print("-" * 50)

print(f"Прогнозований клас для тестової точки: {predicted_label}")
if predicted_label == '>50K':
    print("Тестова точка даних належить до класу: Особа має дохід більше $50K на рік (>50K).")
else:
    print("Тестова точка даних належить до класу: Особа має дохід менше або дорівнює $50K на рік (<=50K).")
print(f"\nОбчислені показники якості класифікації на тестовому наборі:")
print(f"Акуратність (Accuracy): {round(100 * accuracy, 2)}%")
print(f"Точність (Precision): {round(100 * precision, 2)}%")
print(f"Повнота (Recall): {round(100 * recall, 2)}%")
print(f"F1-міра (F1 Score): {round(100 * f1_test, 2)}%")
print("-" * 50)
