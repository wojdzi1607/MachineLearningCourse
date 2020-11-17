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
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    plot_iris_3D(X, y)
    # plt.show()

    clusters = range(2, 15)
    inetries = []
    for n in clusters:
        kmeans = cluster.KMeans(n_clusters=n).fit(X)
        inetries.append(kmeans.inertia_)

    plt.figure()
    plt.plot(clusters, inetries, '-o', color='black')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('inertia_')
    plt.xticks(clusters)
    plt.show()

    kmeans = cluster.KMeans().fit(X)
    kmeans_labels = kmeans.labels_
    plot_iris_3D(X, kmeans_labels)
    # plt.show()

    clustering = cluster.AgglomerativeClustering().fit(X)
    clustering_labels = clustering.labels_
    plot_iris_3D(X, clustering_labels)
    # plt.show()

    db = cluster.DBSCAN().fit(X)
    db_labels = db.labels_
    plot_iris_3D(X, db_labels)
    # plt.show()

    print('\nadjusted_rand_score')
    print(f'clustering: {metrics.adjusted_rand_score(y, clustering_labels)}')
    print(f'kmeans: {metrics.adjusted_rand_score(y, kmeans_labels)}')
    print(f'db: {metrics.adjusted_rand_score(y, db_labels)}')

    print('\ncalinski_harabasz_score')
    print(f'clustering: {metrics.calinski_harabasz_score(X, clustering_labels)}')
    print(f'kmeans: {metrics.calinski_harabasz_score(X, kmeans_labels)}')
    print(f'db: {metrics.calinski_harabasz_score(X, db_labels)}')


def TODO2():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    pca = PCA(n_components=2)
    pca.fit(X)
    X_c = pca.transform(X)
    plot_iris(X_c, y)
    plt.show()


def main():
    TODO2()


if __name__ == '__main__':
    main()
