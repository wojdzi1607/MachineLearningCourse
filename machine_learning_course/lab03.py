import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble

from mlxtend.plotting import plot_decision_regions


def plot_iris(X: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')


def TODO1():
    iris = datasets.load_iris()
    # iris = datasets.load_iris(as_frame=True)
    # print(iris.frame.describe())

    X, y = iris.data, iris.target
    print(f'count y : {np.bincount(y)}')

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y)

    print(f'county y_train: {np.bincount(y_train)}')
    print(f'count y_test : {np.bincount(y_test)}')

    plot_iris(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_min_max_scaler = min_max_scaler.transform(X)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_standard_scaler = scaler.transform(X)

    plot_iris(X_min_max_scaler)
    plot_iris(X_standard_scaler)
    plt.show()


def TODO2():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X = X[:, [0, 1]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(X_train)

    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # classificators
    clf_svm = svm.SVC(random_state=42)
    clf_svm.fit(X_train, y_train)
    acc_clf_svm = metrics.accuracy_score(y_test, clf_svm.predict(X_test))
    print(f'acc_clf_svm: {acc_clf_svm}')

    clf_linear = linear_model.LogisticRegression(random_state=42)
    clf_linear.fit(X_train, y_train)
    acc_clf_linear = metrics.accuracy_score(y_test, clf_linear.predict(X_test))
    print(f'acc_clf_linear: {acc_clf_linear}')

    clf_tree = tree.DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train, y_train)
    acc_clf_tree = metrics.accuracy_score(y_test, clf_tree.predict(X_test))
    print(f'acc_clf_tree: {acc_clf_tree}')

    clf_rf = ensemble.RandomForestClassifier(random_state=42)
    clf_rf.fit(X_train, y_train)
    acc_clf_rf = metrics.accuracy_score(y_test, clf_rf.predict(X_test))
    print(f'acc_clf_rf: {acc_clf_rf}')

    # plt.figure()
    # plot_decision_regions(X_test, y_test, clf_svm, legend=2)
    # plt.figure()
    # plot_decision_regions(X_test, y_test, clf_linear, legend=2)
    # plt.figure()
    # plot_decision_regions(X_test, y_test, clf_tree, legend=2)
    # plt.figure()
    # plot_decision_regions(X_test, y_test, clf_rf, legend=2)
    #
    # plt.show()

    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    clf_gs = model_selection.GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, n_jobs=4)
    clf_gs.fit(X_train, y_train)
    print(clf_gs.cv_results_)


def main():
    TODO2()


if __name__ == '__main__':
    main()
