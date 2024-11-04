import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import kMeans_method, dendrogram_method, dbscan_method, spectral_method
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
            s, db, ch = dendrogram_method(count_clusters, X)
            self.ui.labelDendSil.setText(str(s))
            self.ui.labelDendDav.setText(str(db))
            self.ui.labelDendCal.setText(str(ch))

        if self.ui.dbscanRadioButton.isChecked():
            s, db, ch = dbscan_method(X)
            self.ui.labelDBScanSil.setText(str(s))
            self.ui.labelDBScanDav.setText(str(db))
            self.ui.labelDBScanCal.setText(str(ch))

        if self.ui.spectralRadioButton.isChecked():
            s, db, ch = spectral_method(count_clusters, X)
            self.ui.labelSpecSil.setText(str(s))
            self.ui.labelSpecDav.setText(str(db))
            self.ui.labelSpecCal.setText(str(ch))
            
        if self.ui.kmeansRadioButton.isChecked():
            s, db, ch = kMeans_method(count_clusters, X)
            self.ui.labelKMSil.setText(s)
            self.ui.labelKMDav.setText(db)
            self.ui.labelKMCal.setText(ch)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
