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
from sklearn.decomposition import PCA
from sklearn import cluster
from mlxtend.plotting import plot_decision_regions
from sklearn.impute import SimpleImputer


def plot_iris(X: np.ndarray, y: np.ndarray) -> None:
    # Wizualizujemy tylko dwie pierwsze cechy – aby móc je przedstawić bez problemu w 2D.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.axvline(x=0)
    plt.axhline(y=0)
    plt.title('Iris sepal features')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('sepal width (cm)')


def plot_iris_3D(X: np.ndarray, y: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 2], X[:, 3], c=y)


def TODO1():
    X, y = datasets.fetch_openml('diabetes', as_frame=True, return_X_y=True)

    print(y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    isolation_forest = ensemble.IsolationForest(contamination='auto')
    isolation_forest.fit(X_train)
    y_predicted_outliers = isolation_forest.predict(X_test)
    print(y_predicted_outliers)
    plot_iris(X_test.values, y_predicted_outliers)

    clf_x = svm.SVC()
    clf_x.fit(X_train, y_train)
    SimpleImputer(missing_values=0.0)
    y_pred = clf_x.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    clf_rf = ensemble.RandomForestClassifier()
    clf_rf.fit(X_train, y_train)

    importances = clf_rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # X_train.boxplot()
    # plt.show()

    # y[y == 'tested_negative'] = 0
    # y[y == 'tested_positive'] = 1


def main():
    TODO1()


if __name__ == '__main__':
    main()
