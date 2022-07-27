from PyQt5 import QtWidgets, QtGui, QtCore

from PyQt5.QtWidgets import QMessageBox, QButtonGroup

import sys, subprocess

from vista_audio import Ui_MainWindow

from record_audio import Audio_Creation

from NN_Clase import NN_audio

from cargar_modelo import Nueva_prediccion

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
        self.ui.pushButton_Acelera.clicked.connect(self.boton_Acelera)
        self.ui.pushButton_Disminuye.clicked.connect(self.boton_Disminuye)
        self.ui.pushButton_Luz_Alta.clicked.connect(self.boton_Luz_Alta)
        self.ui.pushButton_Luz_Baja.clicked.connect(self.boton_Luz_Baja)
        self.ui.pushButton_Intermitentes.clicked.connect(self.boton_Intermitentes)
        self.ui.pushButton_Iniciar.clicked.connect(self.boton_Iniciar)

        self.ui.pushButton_prueba.clicked.connect(self.boton_prueba)
        self.ui.pushButton_evaluar.clicked.connect(self.boton_evaluar)
        
        self.ui.lineEdit.setValidator(QtGui.QIntValidator())

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
    
    def boton_Acelera(self):
        a = Audio_Creation("Acelera")
        a.iniciar_programa()
    
    def boton_Disminuye(self):
        a = Audio_Creation("Disminuye")
        a.iniciar_programa()
    
    def boton_Luz_Alta(self):
        a = Audio_Creation("Luz alta")
        a.iniciar_programa()
    
    def boton_Luz_Baja(self):
        a = Audio_Creation("Luz baja")
        a.iniciar_programa()
    
    def boton_Intermitentes(self):
        a = Audio_Creation("Intermitentes")
        a.iniciar_programa()

    def boton_prueba(self):
        a = Audio_Creation("prueba")
        a.iniciar_programa()

    def boton_evaluar(self):
        a = Nueva_prediccion()
        a.iniciar()
        self.ui.label_5.setText(a.resultado)

    def boton_Iniciar(self):
        iteraciones = int(self.ui.lineEdit.text())
        print(iteraciones)

        clase = NN_audio(iteraciones)

        clase.iniciar()

        

app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())