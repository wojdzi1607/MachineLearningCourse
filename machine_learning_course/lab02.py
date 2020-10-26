import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
import numpy as np


def TODO1():
    from sklearn.tree import DecisionTreeClassifier

    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]
    y = [0, 1, 1, 1]

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.
    print(clf.predict(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
    ))

    tree.plot_tree(clf, filled=True)
    plt.show()


def TODO2():
    dict_brand = {'VW': 0, 'Ford': 1, 'Opel': 2}
    dict_changed = {'tak': 0, 'nie': 1}

    # marka, przebieg, czy uszkodzony
    X = [
        ['Opel', 250000, 'tak'],
        ['Opel', 50000, 'nie'],
        ['Opel', 100000, 'tak'],
        ['VW', 5000, 'tak'],
        ['Ford', 200000, 'nie'],
        ['VW', 400000, 'nie']
    ]

    for x in X:
        x[0] = dict_brand[x[0]]
        x[2] = dict_changed[x[2]]

    y = [
        0,
        1,
        0,
        1,
        1,
        0
    ]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    # Sprawdź sam(a) jakie będą wyniki dla innych danych wejściowych.
    print(clf.predict(
        [
           [dict_brand['Opel'], 100000, dict_changed['nie']]
        ]
    ))

    tree.plot_tree(
        clf,
        filled=True,
        feature_names=['marka', 'przebieg', 'uszkodzony'],
        class_names=['nie kupujemy', 'kupujemy']
    )
    plt.show()


def TODO3():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(digits.data,
                                                                        digits.target,
                                                                        test_size=0.33,
                                                                        random_state=42)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    print(f'Confusion matrix:\n{metrics.confusion_matrix(y_predicted, y_test)}')

    metrics.plot_confusion_matrix(clf, X_test, y_test)
    plt.show()


def print_regressor_score(y_test: np.ndarray, y_predicted: np.ndarray) -> None:
    print(f'mae: {metrics.mean_absolute_error(y_test, y_predicted)}')
    print(f'mse: {metrics.mean_squared_error(y_test, y_predicted)}')
    print(f'r2: {metrics.r2_score(y_test, y_predicted)}')


def TODO4():
    data = np.loadtxt('./battery_problem_data.csv', delimiter=',')
    # print(data)

    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    decision_tree_regressor = tree.DecisionTreeRegressor()
    decision_tree_regressor.fit(X_train, y_train)
    y_predicted_decision_tree = decision_tree_regressor.predict(X_test)

    linear_model_regressor = LinearRegression()
    linear_model_regressor.fit(X_train, y_train)
    y_predicted_linear = linear_model_regressor.predict(X_test)

    print_regressor_score(y_test, y_predicted_decision_tree)
    print_regressor_score(y_test, y_predicted_linear)

    plt.scatter(X_test, y_test, c='red', marker='*')
    plt.scatter(X_test, y_predicted_linear, c='green', marker='o')
    plt.scatter(X_test, y_predicted_decision_tree, c='blue', marker='x')

    plt.show()


def main():
    TODO4()


if __name__ == '__main__':
    main()
