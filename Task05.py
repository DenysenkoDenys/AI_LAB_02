import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

print("=" * 70)
print("ЗАВДАННЯ 2.5: КЛАСИФІКАЦІЯ ДАНИХ ЛІНІЙНИМ КЛАСИФІКАТОРОМ RIDGE")
print("=" * 70)

iris = load_iris()
X, y = iris.data, iris.target

print("\n1. ЗАВАНТАЖЕННЯ ТА РОЗДІЛЕННЯ ДАНИХ")
print(f"Форма датасету: {X.shape}")
print(f"Кількість класів: {len(np.unique(y))}")
print(f"Назви класів: {iris.target_names}")

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

print(f"Розмір навчальної вибірки: {Xtrain.shape}")
print(f"Розмір тестової вибірки: {Xtest.shape}")

print("\n2. НАЛАШТУВАННЯ ТА НАВЧАННЯ КЛАСИФІКАТОРА RIDGE")
print("Параметри класифікатора:")
print("  - tol = 1e-2 (точність: 0.01)")
print("  - solver = 'sag' (Stochastic Average Gradient)")
print("  - alpha = 1.0 (за замовчуванням, регуляризація)")

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)

print("\nМодель успішно навчена!")

ypred = clf.predict(Xtest)

print("\n" + "=" * 70)
print("3. ПОКАЗНИКИ ЯКОСТІ КЛАСИФІКАЦІЇ")
print("=" * 70)

accuracy = np.round(metrics.accuracy_score(ytest, ypred), 4)
precision = np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4)
recall = np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4)
f1 = np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4)
cohen_kappa = np.round(metrics.cohen_kappa_score(ytest, ypred), 4)
matthews = np.round(metrics.matthews_corrcoef(ytest, ypred), 4)

print(f'Accuracy (Точність):          {accuracy}')
print(f'Precision (Прецизійність):    {precision}')
print(f'Recall (Повнота):             {recall}')
print(f'F1 Score (F1-міра):           {f1}')
print(f'Cohen Kappa Score:            {cohen_kappa}')
print(f'Matthews Corrcoef:            {matthews}')

print("\n" + "=" * 70)
print("4. ЗВІТ ПРО КЛАСИФІКАЦІЮ")
print("=" * 70)
print(metrics.classification_report(ytest, ypred, target_names=iris.target_names))

print("=" * 70)
print("5. МАТРИЦЯ ПЛУТАНИНИ (CONFUSION MATRIX)")
print("=" * 70)

mat = confusion_matrix(ytest, ypred)
print("Матриця плутанини:")
print(mat)

plt.figure(figsize=(8, 6))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('True Label (Справжня мітка)')
plt.ylabel('Predicted Label (Прогнозована мітка)')
plt.title('Confusion Matrix - Ridge Classifier')
plt.savefig("Confusion.jpg", dpi=300, bbox_inches='tight')
print("\nМатрицю плутанини збережено у файл: Confusion.jpg")

f = BytesIO()
plt.savefig(f, format="svg")
print("Матрицю також збережено у форматі SVG")

plt.show()

print("\n" + "=" * 70)
print("ПРОГРАМУ ВИКОНАНО УСПІШНО!")
print("=" * 70)