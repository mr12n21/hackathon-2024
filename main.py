import pandas as pd
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def load_graph(csv_file):
    data = pd.read_csv(csv_file)
    G = nx.from_pandas_edgelist(data, 'Column0', 'Column1')
    adj_matrix = nx.to_numpy_array(G)
    return adj_matrix

def pad_matrix(matrix, target_size):
    padded_matrix = np.zeros((target_size, target_size))
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

files = ['./csv/test.csv', './csv/test1.csv']
adj_matrices = [load_graph(file) for file in files]

for i, matrix in enumerate(adj_matrices):
    print(f"Tvar matice grafu {i} před paddingem: {matrix.shape}")

max_size = max(matrix.shape[0] for matrix in adj_matrices)

adj_matrices_padded = [pad_matrix(matrix, max_size) for matrix in adj_matrices]

for i, matrix in enumerate(adj_matrices_padded):
    print(f"Tvar matice grafu {i} po paddingu: {matrix.shape}")

X = np.array(adj_matrices_padded)
X = X / np.max(X)

X_flattened = X.reshape(X.shape[0], -1)
print(f"Tvar vektorizovaných dat: {X_flattened.shape}")

input_dim = X_flattened.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

autoencoder.fit(X_flattened, X_flattened, epochs=100, batch_size=8, shuffle=True)

encoder = Model(inputs=autoencoder.input, outputs=encoded)
latent_vectors = encoder.predict(X_flattened)

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(latent_vectors)

clusters = kmeans.labels_
print(f"Cluster labels: {clusters}")

for i, cluster in enumerate(clusters):
    print(f"Graf {files[i]} je v clusteru {cluster}")