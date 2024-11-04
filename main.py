import sys
import pandas as pd
from model import kMeans_method, dendrogram_method, dbscan_method, spectral_method, draw_silhouette
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore, QtWidgets, uic
from GUIforClustering import Ui_MainWindow  # Убедитесь, что GUIforClustering находится в том же каталоге

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


# Функция для кнопки коэффициента силуэта
def silhouette_button():
    data = pd.read_csv(r'E:/PythonDinarDiploma/cars.csv')  # Убедитесь, что путь к файлу корректный
    X = data[['Цена', 'Пробег', 'Год', 'Владельцы']]
    draw_silhouette(X)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
