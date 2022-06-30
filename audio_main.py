from PyQt5 import QtWidgets, QtGui, QtCore

from PyQt5.QtWidgets import QMessageBox, QButtonGroup

import sys, subprocess

from vista_audio import Ui_MainWindow

from record_audio import Audio_Creation


class mywindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        mywindow.setWindowTitle(self,"Programa de grabación de palabras clave")

        self.ui.pushButton_Adelante.clicked.connect(self.boton_Adelante)
        self.ui.pushButton_Atras.clicked.connect(self.boton_Atras)
        self.ui.pushButton_Izquierda.clicked.connect(self.boton_Izquierda)
        self.ui.pushButton_Derecha.clicked.connect(self.boton_Derecha)
        self.ui.pushButton_Alto.clicked.connect(self.boton_Alto)


    def boton_Adelante(self):
        a = Audio_Creation("Adelante")
        a.iniciar_programa()
    
    def boton_Atras(self):
        a = Audio_Creation("Atras")
        a.iniciar_programa()

    def boton_Izquierda(self):
        a = Audio_Creation("Izquierda")
        a.iniciar_programa()

    def boton_Derecha(self):
        a = Audio_Creation("Derecha")
        a.iniciar_programa()

    def boton_Alto(self):
        a = Audio_Creation("Alto")
        a.iniciar_programa()

        

app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())