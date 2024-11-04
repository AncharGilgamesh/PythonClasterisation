import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore, QtWidgets, uic
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, pairwise_distances, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from GUIforClustering import Ui_MainWindow  # Убедитесь, что GUIforClustering находится в том же каталоге

# Функция для кнопки коэффициента силуэта
def silhouette_button():
    data = pd.read_csv(r'E:/PythonDinarDiploma/cars.csv')  # Убедитесь, что путь к файлу корректный
    X = data[['Цена', 'Пробег', 'Год', 'Владельцы']]
    silhouette_scores = []

    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg:.2f}")

    # Визуализация коэффициента силуэта
    plt.plot(range(2, 11), silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Method For Optimal k')
    plt.show()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Подключение кнопок
        self.ui.labelOcenka.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.labelQuantity.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.spinBox.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.result_button.clicked.connect(self.buttonClicked)
        self.ui.silhouette_button.clicked.connect(silhouette_button)

    def buttonClicked(self):
        count_clusters = self.ui.spinBox.value()
        data = pd.read_csv(r'E:/PythonDinarDiploma/cars.csv')
        X = data[['Цена', 'Пробег', 'Год', 'Владельцы']]

        if self.ui.dendrogramRadioButton.isChecked():
            silhouette_scores = []
            for n_clusters in range(2, 11):
                kmeans = KMeans(n_clusters=n_clusters)
                cluster_labels = kmeans.fit_predict(X[['Пробег', 'Цена']])
                silhouette_avg = silhouette_score(X[['Пробег', 'Цена']], cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f'For n_clusters  = {n_clusters}, the average silhouette score is {silhouette_avg:.2f}')
            # Визуализация коэффициента сиулэта
            plt.plot(range(2,11), silhouette_scores, 'bo-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette score')
            plt.title('Silhouette Method For Optimal')
            plt.show()
            Z_centroid = linkage(X[['Цена', 'Пробег', 'Год', 'Владельцы']], 
            method='centroid', metric='euclidean')
            # count_clusters = 3
            # Вычисление коэффициента силуэтта
            labels = fcluster(Z_centroid, t=count_clusters, criterion='maxclust')
            # Создание нового графика
            plt.figure(figsize=(8, 6))

            distances = pairwise_distances(X)
            kalinski_harabash =calinski_harabasz_score(X,labels)
            print(kalinski_harabash)
            # Расчет индекса Дэвиса-Болдуина
            davies_bouldin_index = davies_bouldin_score(X, labels)
            print(f"Индекс Дэвиса-Болдуина: {davies_bouldin_index}")
             # Индекс Силуэта
            dbscan_silhouette = silhouette_score(X, labels)
            print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")

            s = str(dbscan_silhouette)
            db = str(davies_bouldin_index)
            ch = str(kalinski_harabash)
            self.ui.labelDendSil.setText(str(s))
            self.ui.labelDendDav.setText(str(db))
            self.ui.labelDendCal.setText(str(ch))
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

        if self.ui.dbscanRadioButton.isChecked():
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
            self.ui.labelDBScanSil.setText(str(dbscan_silhouette))
            self.ui.labelDBScanDav.setText(str(davies_bouldin_index))
            self.ui.labelDBScanCal.setText(str(kalinski_harabash))

            # График кластеров
            plt.scatter(data['Цена'], data['Пробег'], c=labels, s=data['Год'] * 0.01, marker='o', alpha=0.5)
            plt.xlabel('Цена')
            plt.ylabel('Пробег')
            plt.title('Кластеризация методом DBSCAN')
            plt.show()
        if self.ui.spectralRadioButton.isChecked():
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
            self.ui.labelSpecSil.setText(str(silhouette_avg))
            self.ui.labelSpecDav.setText(str(davies_bouldin_avg))
            self.ui.labelSpecCal.setText(str(calinski_harabasz_avg))
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
            


        if self.ui.kmeansRadioButton.isChecked():
            # Создание объекта KMeans и кластеризация
            kmeans = KMeans(n_clusters=count_clusters)
            kmeans.fit(X)
            # Индекс Силуэта
            labels = kmeans.predict(X)
            kmeans_labels = kmeans.labels_
            kmeans_silhouette = silhouette_score(X, kmeans_labels)
            distances = pairwise_distances(X)
            # Расчет индекса Дэвиса-Болдуина
            davies_bouldin_index = davies_bouldin_score(X, labels)
            kalinski_harabash = calinski_harabasz_score(X, labels)
            print(f"Индекс Дэвиса-Болдуина: {davies_bouldin_index}")
            print(f"K-Means Silhouette Score: {kmeans_silhouette}")
            s = str(kmeans_silhouette)
            db = str(davies_bouldin_index)
            ch = str(kalinski_harabash)
            self.ui.labelKMSil.setText(s)
            self.ui.labelKMDav.setText(db)
            self.ui.labelKMCal.setText(ch)
            # визуализация кластеров
            plt.scatter(X['Цена'], X['Пробег'], c=labels,  s=X['Год'] * 0.01, marker='o', alpha=0.5)
            plt.xlabel('Цена')
            plt.ylabel('Пробег')
            plt.title('Кластеризация методом k-средних')
            plt.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
