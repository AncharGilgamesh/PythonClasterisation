import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, pairwise_distances, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

def kMeans_method(count_clusters, X):
    # Создание объекта KMeans и кластеризация
    kmeans = KMeans(n_clusters=count_clusters)
    kmeans.fit(X)
    # Индекс Силуэта
    labels = kmeans.predict(X)
    kmeans_labels = kmeans.labels_
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    #distances = pairwise_distances(X)
    # Расчет индекса Дэвиса-Болдуина
    davies_bouldin_index = davies_bouldin_score(X, labels)
    kalinski_harabash = calinski_harabasz_score(X, labels)

    s = str(kmeans_silhouette)
    db = str(davies_bouldin_index)
    ch = str(kalinski_harabash)

    # визуализация кластеров
    plt.scatter(X['Цена'], X['Пробег'], c=labels,  s=X['Год'] * 0.01, marker='o', alpha=0.5)
    plt.xlabel('Цена')
    plt.ylabel('Пробег')
    plt.title('Кластеризация методом k-средних')
    plt.show()
    return s, db, ch

def spectral_method(count_clusters, X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    silhouette_scores = []
    for n_clusters in range(2, 6):
        model = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=0)
        cluster_labels = model.fit_predict(X_std)
        silhouette_avg = silhouette_score(X_std, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    davies_bouldin_avg = davies_bouldin_score(X_std, cluster_labels)
    calinski_harabasz_avg = calinski_harabasz_score(X_std, cluster_labels)
    # Построение графика
    plt.plot(range(2, 6), silhouette_scores)
    plt.xlabel('Количество кластеров')
    plt.ylabel('Коэффициент силуэта')
    plt.show()

    # Кластеризация с оптимальным количеством кластеров
    best_n_clusters = count_clusters
    model = SpectralClustering(n_clusters=best_n_clusters, affinity='rbf', random_state=0)
    labels = model.fit_predict(X_std)

    # Визуализация кластеров
    plt.scatter(X['Цена'], X['Пробег'], c=labels)
    plt.xlabel('Цена')
    plt.ylabel('Пробег')
    plt.title(f'Спектральная кластеризация с делением на {best_n_clusters} кластера')
    plt.show()
    return silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg

def dbscan_method(X):
        # Стандартизация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Кластеризация DBSCAN
    dbscan = DBSCAN(eps=0.7, min_samples=4)
    labels = dbscan.fit_predict(X_scaled)

    # Метрики качества
    dbscan_silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin_index = davies_bouldin_score(X_scaled, labels)
    kalinski_harabash = calinski_harabasz_score(X_scaled, labels)

    # Вывод метрик
    #self.ui.labelDBScanSil.setText(str(dbscan_silhouette))
    #self.ui.labelDBScanDav.setText(str(davies_bouldin_index))
    #self.ui.labelDBScanCal.setText(str(kalinski_harabash))

    # График кластеров
    plt.scatter(X['Цена'], X['Пробег'], c=labels, s=X['Год'] * 0.01, marker='o', alpha=0.5)
    plt.xlabel('Цена')
    plt.ylabel('Пробег')
    plt.title('Кластеризация методом DBSCAN')
    plt.show()
    return dbscan_silhouette, davies_bouldin_index, kalinski_harabash

def dendrogram_method(count_clusters, X):
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(X[['Пробег', 'Цена']])
        silhouette_avg = silhouette_score(X[['Пробег', 'Цена']], cluster_labels)
        silhouette_scores.append(silhouette_avg)
    # Визуализация коэффициента сиулэта
    Z_centroid = linkage(X[['Цена', 'Пробег', 'Год', 'Владельцы']], 
    method='centroid', metric='euclidean')
    # Вычисление коэффициента силуэтта
    labels = fcluster(Z_centroid, t=count_clusters, criterion='maxclust')
    # Создание нового графика
    plt.figure(figsize=(8, 6))

    #distances = pairwise_distances(X)
    kalinski_harabash =calinski_harabasz_score(X,labels)
    # Расчет индекса Дэвиса-Болдуина
    davies_bouldin_index = davies_bouldin_score(X, labels)
     # Индекс Силуэта
    dbscan_silhouette = silhouette_score(X, labels)

    s = str(dbscan_silhouette)
    db = str(davies_bouldin_index)
    ch = str(kalinski_harabash)
    # Разбиение точек на кластеры и вывод их в виде точек на графике
    plt.scatter(X['Цена'], X['Пробег'], c=labels, cmap='viridis')
    # Добавление меток для осей
    plt.title('Centroid method')
    plt.xlabel('Цена')
    plt.ylabel('Пробег')
    # Вывод графика
    plt.show()
    plt.figure(figsize=(12, 6))
    dendrogram(Z_centroid, color_threshold=380752)
    plt.xticks(rotation=90)
    plt.title('Centroid linkage')
    plt.show()
    return s, db, ch

def draw_silhouette(X):
    silhouette_scores = []

    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Визуализация коэффициента силуэта
    plt.plot(range(2, 11), silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()