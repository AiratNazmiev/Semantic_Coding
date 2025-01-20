import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_visualization(X_train, X_test, labels_test):
    pca = PCA(n_components=2)
    _ = pca.fit_transform(X_train)
    X_test_latent_pca = pca.transform(X_test)


    plt.figure(figsize=(10, 6), dpi=300)
    for i in range(10):
        num_mask = (labels_test == i)
        plt.scatter(X_test_latent_pca[num_mask, 0], X_test_latent_pca[num_mask, 1], s=1, color=f'C{i}', label=f'{i}');
    plt.legend()
    plt.grid()
    plt.show()