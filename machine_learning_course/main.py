import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split


def TODO1():
    # requirements.txt
    pass


def TODO2():
    digits = datasets.load_digits()
    # print(digits)

    clf = svm.SVC()
    print(clf.fit(digits.data[:-1], digits.target[:-1]))
    print(clf.predict([digits.data[-1]]))

    pickle.dump(clf, open('./clf.p', 'wb'))
    clf = pickle.load(open('./clf.p', 'rb'))

    if clf.predict([digits.data[-1]]) == digits.target[-1]:
        print('Poprawność 100%')
    else:
        print('Poprawność 0%')


def TODO3():
    digits = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.7, random_state=42)
    print('X_train: ', X_train)
    print('X_test: ', X_test)
    print('y_train: ', y_train)
    print('y_test: ', y_test)


def TODO4():
    faces = datasets.fetch_olivetti_faces()
    print(faces)

    print(faces.DESCR)
    print(faces.data)
    print(faces.target)

    X, y = datasets.fetch_olivetti_faces(return_X_y=True)


def TODO5():
    boston = datasets.load_boston()
    print(boston.DESCR)
    print(boston.data)
    print(boston.target)
    print(boston['feature_names'])


def TODO6():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=3,
        n_informative=2, n_redundant=0, n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.0
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def TODO7():
    d = datasets.fetch_openml(data_id=40536, as_frame=True)
    print(d)


def TODO8():
    def regressor(x: float) -> float:
        if x >= 4.0:
            return 8.0
        else:
            return 2*x

    data = np.loadtxt('./battery_problem_data.csv', delimiter=',')
    print(data)

    X = data[:, 0]
    y = data[:, 1]

    y_predicted = []
    for single_data in X:
        y_predicted.append(regressor(single_data))

    plt.scatter(X, y)
    plt.scatter(X, y_predicted, marker='*', c='red')
    plt.show()


def main():
    TODO8()


if __name__ == '__main__':
    main()
