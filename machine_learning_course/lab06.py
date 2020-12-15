import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def TODO1():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    random.seed(42)

    X, y = datasets.fetch_openml(name='Titanic', version=1, return_X_y=True, as_frame=True)
    X: pd.DataFrame = X

    print(X.head(5))
    print(X.info())
    print(X.describe())

    X.drop(['boat', 'body', 'home.dest'], axis=1, inplace=True)

    # PREPROCESSING
    X = X.drop(columns=['cabin', 'embarked', 'age', 'name'])
    X['sex'].replace({'male': 0, 'female': 1}, inplace=True)
    y = y.drop(columns=['cabin', 'embarked', 'age'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print(X_train.info())

    y_predict_random = random.choices(['0', '1'], k=len(y_test))
    print(metrics.classification_report(y_test, y_predict_random))

    y_predict_0 = ['0']*len(y_test)
    print(metrics.classification_report(y_test, y_predict_0))

    # UZUPEŁNIENIE DBRAKUJĄCYCH ANYCH

    X_combined = pd.concat([X_train, y_train.astype(float)], axis=1)
    print(X_combined.head(5))

    df_tmp = X_combined[['sex', 'survived']].groupby('sex').mean()
    print(df_tmp.head(5))

    df_tmp = X_combined[['pclass', 'survived']].groupby('pclass').mean()
    print(df_tmp.head(5))

    # X_combined['sex'].replace({'male': 0, 'female': 1}, inplace=True)

    print(X_combined.corr())
    sns.heatmap(X_combined.corr(), annot=True, cmap='coolwarm')
    plt.show()


def TODO2():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    random.seed(42)

    X, y = datasets.fetch_openml(name='Titanic', version=1, return_X_y=True, as_frame=True)
    X: pd.DataFrame = X

    # PREPROCESSING
    X['sex'].replace({'male': 0, 'female': 1}, inplace=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)
    X = X.drop(columns=['age', 'body'])
    X = X.fillna(method='bfill', axis=0).fillna(0)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42,
                                                                        stratify=y)
    print(X_train.info())
    print(X_test.info())

    # UCZENIE MASZYNOWE
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f'mean_absolute_error: {mean_absolute_error(y_test, preds)}')
    print(f'model.score: {model.score(X_test, y_test)}')


def main():
    TODO2()


if __name__ == '__main__':
    main()
