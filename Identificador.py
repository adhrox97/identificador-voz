"""
	Tomar foto con Python y opencv
	@date 20-03-2018
	@author parzibyte
	@see https://www.parzibyte.me/blog
"""
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from wav_rw import wavread
from scipy.fftpack import fft
from python_speech_features import mfcc
from pandas import Series
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys
import pickle
import wave
import cv2

# def vad(logE,trh):
#     naverage = 10
#     s=Series(logE)
#     logE = (s.rolling(window=naverage)).mean().values
#     logE[0:naverage-1]=logE[naverage:].min()
#     logE=np.roll(logE, -int(naverage/2), axis=0)    
#     wvad=logE>trh
#     return wvad

nombre=''
programa=''
GS=''
ID=''
img=''

class VentanaPrincipal(QMainWindow):

    def __init__(self, parent=None):

        super(VentanaPrincipal, self).__init__(parent)
        self.MainWindow=loadUi('ventanaprincipal.ui', self)
        self.botongr.clicked.connect(self.grabarvoz)

    def grabarvoz(self):

        global nombre
        global programa
        global GS
        global ID
        global img
        #DEFINIMOS PARAMETROS
        FORMAT=pyaudio.paInt16
        CHANNELS=1
        RATE=16000
        CHUNK=1024
        duracion=4
        archivo="inTime.wav"

        #INICIAMOS "pyaudio"
        audio=pyaudio.PyAudio()

        #INICIAMOS GRABACIÓN
        stream=audio.open(format=FORMAT,channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        print("grabando...")

        frames=[]

        for i in range(0, int(RATE/CHUNK*duracion)):
            self.label1.setText('grabando')
            data=stream.read(CHUNK)
            frames.append(data)
        print("grabación terminada")

        #DETENEMOS GRABACIÓN
        stream.stop_stream()
        stream.close()
        audio.terminate()

        #CREAMOS/GUARDAMOS EL ARCHIVO DE AUDIO
        waveFile = wave.open(archivo, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()


        (r,s)=wavread(archivo)
        MFCC=mfcc(s,samplerate=16000,winlen=0.025,winstep=0.01,numcep=20,nfilt=27,nfft=512,lowfreq=0,preemph=0.97,ceplifter=22,appendEnergy=True,winfunc=np.hamming)
        cova=np.cov(MFCC,rowvar=False)
        vecc=cova[np.triu_indices(13, k = 0)]
        C=vecc
        # plot data
        #plt.subplot(211)
        #plt.subplot(212)
        #plt.plot(numpydata)
        C=C/(abs(C).max())

        plt.plot(C)
        clf = pickle.load(open("SVM.p","rb"))
        decision=clf.predict(C.reshape(1,-1))
        print("se dijo",decision)
        data=0
        numpydata=0
        C=0
        MFCC=0
        stream=0
        p=0
        clf=0
        s=0


        if(decision==1):

            nombre='ADRIAN DAVID CEBALLOS MURILLO'
            programa='INGENIERIA ELECTRONICA'
            GS='O+'
            ID='1107513461'
            img='Identidades/adrian.jpg'
            self.abrirVentanaIdentificacion()
            
        elif(decision==2):

            nombre='JUAN DAVID CAÑARTE MARTINEZ'
            programa='INGENIERIA ELECTRONICA'
            GS='O+'
            ID='98011456847'
            img='Identidades/canarte.jpg'
            self.abrirVentanaIdentificacion()

        elif(decision==3):
            
            nombre='JUAN JOSE RODRIGUEZ RODRIGUEZ'
            programa='INGENIERIA ELECTRONICA'
            GS='O+'
            ID='97647842567'
            img='Identidades/Juan.jpg'
            self.abrirVentanaIdentificacion()

       # plt.show()

    def abrirVentanaIdentificacion(self):

        self.close()
        otraventana=VentanaIdentificacion(self)
        otraventana.show()

class VentanaIdentificacion(QMainWindow):

    def __init__(self, parent=None):

        global nombre
        global programa
        global GS
        global ID
        global img
        super(VentanaIdentificacion, self).__init__(parent)
        self.MainWindow=loadUi('ventanaidentificacion.ui', self)
        self.b_atras.clicked.connect(self.abrirVentanaPrincipal)

        self.label_1.setText('<html><head/><body><p><span style=" font-weight:600;">NOMBRE</span></p></body></html>\n'+ nombre); 
        self.label_2.setText('<html><head/><body><p><span style=" font-weight:600;">PROGRAMA</span></p></body></html>\n'+ programa)
        self.label_3.setText('<html><head/><body><p><span style=" font-weight:600;">IDENTIFICACION</span></p></body></html>\n'+ ID)
        self.label_4.setText('<html><head/><body><p><span style=" font-weight:600;">G.S.</span></p></body></html>\n'+ GS)

        frame=cv2.imread(img, cv2.IMREAD_COLOR)

        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.shape[1] * frame.shape[2], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap()
        pixmap.convertFromImage(image.rgbSwapped())
        self.MainWindow.labelfoto.setPixmap(pixmap)

    def abrirVentanaPrincipal(self):

        self.close()
        otraventana=VentanaPrincipal(self)
        otraventana.show()

app = QApplication(sys.argv)
main = VentanaPrincipal()
main.show()
sys.exit(app.exec_())