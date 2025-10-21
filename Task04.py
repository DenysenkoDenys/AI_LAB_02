import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

input_file = 'income_data.txt'
max_datapoints = 5000

X_raw = []
y_raw = []
count_class1 = 0
count_class2 = 0

print("=" * 50)
print("1. ЗАВАНТАЖЕННЯ ДАНИХ З income_data.txt ТА ПОПЕРЕДНЯ ОБРОБКА")
print("=" * 50)

try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break

            if '?' in line:
                continue

            data = line[:-1].split(', ')

            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X_raw.append(data[:-1])
                y_raw.append(data[-1])
                count_class1 += 1

            elif data[-1] == '>50K' and count_class2 < max_datapoints:
                X_raw.append(data[:-1])
                y_raw.append(data[-1])
                count_class2 += 1
except FileNotFoundError:
    print(f"ПОМИЛКА: Файл '{input_file}' не знайдено. Переконайтеся, що він знаходиться у робочій директорії.")
    exit()
except Exception as e:
    print(f"Виникла помилка під час читання файлу: {e}")
    exit()

X_raw = np.array(X_raw)
y_raw = np.array(y_raw)

print(f"Завантажено прикладів: {len(X_raw)} (<=50K: {count_class1}, >50K: {count_class2})")

label_encoder_list = []
X_encoded = np.empty(X_raw.shape)

for i in range(X_raw.shape[1]):
    is_numeric = np.char.isdigit(X_raw[:100, i]).all()

    if is_numeric:
        X_encoded[:, i] = X_raw[:, i].astype(float)
        label_encoder_list.append(None)
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X_raw[:, i])
        label_encoder_list.append(le)

X = X_encoded.astype(float)
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

y_encoder = preprocessing.LabelEncoder()
Y = y_encoder.fit_transform(y_raw)
target_names = y_encoder.classes_

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=1,
    stratify=Y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)

print("\n2. МАСШТАБУВАННЯ ТА РОЗДІЛЕННЯ ДАНИХ")
print(f"Форма навчального набору (X_train): {X_train.shape}")
print(f"Форма контрольного набору (X_validation): {X_validation.shape}")

print("\n" + "=" * 50)
print("3. КЛАСИФІКАЦІЯ (ПОРІВНЯННЯ 6-ТИ АЛГОРИТМІВ)")
print("Оцінка за допомогою 10-кратної стратифікованої крос-валідації (Метрика: Accuracy)")
print("=" * 50)

models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=500, random_state=1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=1)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto', random_state=1)))

results = []
names = []
print("Результати (Назва: Середня Точність (Стандартне відхилення)):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

print("\nВиведення діаграми порівняння алгоритмів. Закрийте вікно, щоб продовжити.")
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Algorithm Comparison (Accuracy) for Income Data')
pyplot.show()

best_model_name = 'SVM'
model = SVC(gamma='auto', random_state=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n" + "=" * 50)
print(f"4. ОЦІНКА НАЙКРАЩОЇ МОДЕЛІ ({best_model_name}) НА КОНТРОЛЬНІЙ ВИБІРЦІ")
print("=" * 50)

print(f"Точність (Accuracy) на контрольній вибірці: {accuracy_score(Y_validation, predictions):.4f}")
print("\nМатриця помилок (Confusion Matrix):")
print(confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію (Classification Report):")
print(classification_report(Y_validation, predictions, target_names=target_names))

print("\n" + "=" * 50)
print("Програму виконано успішно!")
print("=" * 50)