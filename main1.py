import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def analyze_curve(file_path):
    data = pd.read_csv(file_path, header=None)

    data = data.apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)

    x_values = data[0].to_numpy()
    y_values = data[1].to_numpy()

    maxima_indices, _ = find_peaks(y_values)
    maxima = y_values[maxima_indices]
    minima_indices, _ = find_peaks(-y_values)
    minima = y_values[minima_indices]

    mean_value = np.mean(y_values)

    threshold = 0.1 * np.max(maxima)
    similar_height_points = y_values[(y_values >= (np.max(maxima) - threshold)) & (y_values <= np.max(maxima))]

    if len(similar_height_points) == 0:
        similar_height_points = np.array([0])

    data_array = np.concatenate((maxima, minima, [mean_value], similar_height_points))

    max_length = 10
    if len(data_array) < max_length:
        data_array = np.pad(data_array, (0, max_length - len(data_array)), 'constant')

    return data_array

def perform_kmeans(data_array, n_clusters=3):
    X = data_array.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    labels = kmeans.labels_
    return labels, kmeans.cluster_centers_

def plot_results(data_array, labels, cluster_centers):
    plt.figure(figsize=(10, 6))
    
    plt.scatter(range(len(data_array)), data_array, c=labels, cmap='viridis', label='Data Points', alpha=0.6)

    plt.scatter(range(len(cluster_centers)), cluster_centers.flatten(), c='red', marker='X', s=100, label='Cluster Centers')

    plt.xlabel('Čas')
    plt.ylabel('Intenzita výboje')
    plt.colorbar(label='')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_path = 'test4.csv'
    data_array = analyze_curve(file_path)
    labels, cluster_centers = perform_kmeans(data_array)

    plot_results(data_array, labels, cluster_centers)
