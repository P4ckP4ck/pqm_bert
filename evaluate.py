from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from network import create_bert_model
from helper import prepare_data
import numpy as np


if __name__ == '__main__':
    data_raw = np.load("training_data/x_train.npy")
    data = prepare_data(data_raw)

    input_shape = (3, 68)
    batch_size = 2048

    model = create_bert_model(input_shape)
    model.load_weights("bert_pre_train.h5")

    embedding_model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    embeddings = embedding_model.predict(data, batch_size=2048)

    X = np.mean(embeddings, axis=1)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reduced Embedding')

    for i in range(0, 360, 5):
        ax.view_init(elev=30, azim=i)
        plt.savefig(f"{i}.jpg")
