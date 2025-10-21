import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


print("=" * 50)
print("КРОК 1: ЗАВАНТАЖЕННЯ ТА ВИВЧЕННЯ ДАНИХ (Використовуємо load_iris для ознайомлення)")
print("=" * 50)

iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print("\nКороткий опис набору даних (DESCR):")
print(iris_dataset['DESCR'][:193] + "\n...")

print("\nНазви відповідей (цільових класів): {}".format(iris_dataset['target_names']))
print("Назви ознак: \n{}".format(iris_dataset['feature_names']))

print("\nТип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data (прикладів, ознак): {}".format(iris_dataset['data'].shape))
print("Значення ознак для перших п'яти прикладів:\n{}".format(iris_dataset['data'][:5]))

print("\nТип масиву target: {}".format(type(iris_dataset['target'])))
print("Відповіді (цільові мітки):\n{}".format(iris_dataset['target']))
print("Значення 0, 1, 2 відповідають сортам: 0 - setosa, 1 - versicolor, 2 - virginica.")

print("\n" + "=" * 50)
print("Продовжуємо завантаження даних для моделювання з URL (pandas)")
print("=" * 50)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print("\nФорма датасету (екземплярів, атрибутів):")
print(dataset.shape)

print("\nПерші 20 рядків датасету:")
print(dataset.head(20))

print("\nСтатистичні зведення числових атрибутів:")
print(dataset.describe())

print("\nРозподіл екземплярів за класом:")
print(dataset.groupby('class').size())


print("\n" + "=" * 50)
print("КРОК 2: ВІЗУАЛІЗАЦІЯ ДАНИХ")
print("=" * 50)

print("Виведення діаграми розмаху (Box Plot). Закрийте вікно, щоб продовжити.")
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.suptitle('Діаграма розмаху атрибутів вхідних даних')
pyplot.show()

print("Виведення гістограми розподілу (Histogram). Закрийте вікно, щоб продовжити.")
dataset.hist()
pyplot.suptitle('Гістограма розподілу атрибутів датасету')
pyplot.show()

print("Виведення матриці діаграм розсіювання. Закрийте вікно, щоб продовжити.")
scatter_matrix(dataset)
pyplot.suptitle('Матриця діаграм розсіювання')
pyplot.show()


print("\n" + "=" * 50)
print("КРОК 3: СТВОРЕННЯ НАВЧАЛЬНОГО ТА ТЕСТОВОГО НАБОРІВ")
print("=" * 50)

array = dataset.values
X = array[:, 0:4].astype(float)
Y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=1
)

print(f"Форма навчального набору (X_train): {X_train.shape}")
print(f"Форма контрольного набору (X_validation): {X_validation.shape}")

print("\n" + "=" * 50)
print("КРОК 4: КЛАСИФІКАЦІЯ (ПОРІВНЯННЯ 6-ТИ АЛГОРИТМІВ)")
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
pyplot.title('Algorithm Comparison (Accuracy)')
pyplot.show()
best_model_name = 'SVM'
model = SVC(gamma='auto', random_state=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n" + "=" * 50)
print(f"КРОК 6 & 7: ОЦІНКА НАЙКРАЩОЇ МОДЕЛІ ({best_model_name}) НА КОНТРОЛЬНІЙ ВИБІРЦІ")
print("=" * 50)
print(f"Точність (Accuracy) на контрольній вибірці: {accuracy_score(Y_validation, predictions):.4f}")
print("\nМатриця помилок (Confusion Matrix):")
print(confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію (Classification Report):")
print(classification_report(Y_validation, predictions))
print("\n" + "=" * 50)

print("КРОК 8: ЗАСТОСУВАННЯ МОДЕЛІ (SVM) ДЛЯ ПРОГНОЗУВАННЯ НОВОГО ПРИКЛАДУ")
print("=" * 50)

X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = model.predict(X_new)
predicted_label_name = prediction[0]

print("\nПрогноз (кодована мітка): {}".format(prediction))
print("Спрогнозована мітка (назва сорту): {}".format(predicted_label_name))
print("\nКвітка з наданими вимірами належить до класу: " + predicted_label_name.upper())

print("=" * 50)
print("Програму виконано успішно!")
print("=" * 50)