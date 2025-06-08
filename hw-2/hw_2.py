import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

class TitanicSurvivalPredictor:
    def __init__(self, file_path, hidden_layer_sizes=(100, 50), max_iter=500, random_state=42):
        """
        Args:
            file_path (str): Путь к CSV-файлу датасета Titanic.
            hidden_layer_sizes (tuple): Размеры скрытых слоёв нейросети.
            max_iter (int): Максимальное количество итераций для обучения.
            random_state (int): Фиксация случайного состояния для воспроизводимости.
        """
        self.file_path = file_path
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Загрузка и анализ датасета."""
        self.data = pd.read_csv(self.file_path)
        print('Первые 5 строк датасета:')
        print(self.data.head())
        print('\nИнформация о датасете:')
        print(self.data.info())
        print('\nПропущенные значения:')
        print(self.data.isnull().sum())

    def analyze_correlation(self):
        """Анализ корреляции числовых признаков с целевой переменной."""
        # Выбираем числовые столбцы для корреляции
        numeric_data = self.data.select_dtypes(include=[np.number])
        print('\nКорреляция с выживаемостью (Survived):')
        corr_matrix = numeric_data.corr()
        print(corr_matrix['Survived'].sort_values(ascending=False))
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Корреляционная матрица числовых признаков')
        plt.show()

    def preprocess_data(self):
        """Предобработка данных.
        Исключённые столбцы:
        - PassengerId: уникальный идентификатор, не влияет на выживаемость.
        - Name: имена не содержат прямой информации о выживаемости.
        - Ticket: номера билетов неструктурированы и не коррелируют с целевой переменной.
        - Cabin: слишком много пропусков (~77%), обработка усложняет задачу.
        """
        # Исключение столбцов
        self.data = self.data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

        # Обработка пропущенных значений
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])

        # Кодирование категориальных переменных
        self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
        self.data = pd.get_dummies(self.data, columns=['Embarked'], drop_first=True)

        # Проверка типов данных
        print('\nТипы данных после предобработки:')
        print(self.data.dtypes)

        # Разделение на признаки и целевую переменную
        self.X = self.data.drop('Survived', axis=1)
        self.y = self.data['Survived']

        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        # Масштабирование признаков
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """Обучение нейросети."""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(self.X_train, self.y_train)
        print('Модель обучена.')

    def evaluate_model(self):
        """Оценка модели."""
        y_pred = self.model.predict(self.X_test)
        print('\nТочность модели на тестовой выборке:')
        print(f'Accuracy: {accuracy_score(self.y_test, y_pred):.2f}')
        print('\nПодробный отчет о классификации:')
        print(classification_report(self.y_test, y_pred))

    def run(self):
        """Выполнение всех этапов."""
        self.load_data()
        self.analyze_correlation()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()

if __name__ == '__main__':
    predictor = TitanicSurvivalPredictor(file_path='../train.csv')
    predictor.run()