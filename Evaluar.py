import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from wav_rw import wavread
from scipy.fftpack import fft
from python_speech_features import mfcc
from pandas import Series
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle
import wave

def vad(logE,trh):
    naverage = 10
    s=Series(logE)
    logE = (s.rolling(window=naverage)).mean().values
    logE[0:naverage-1]=logE[naverage:].min()
    logE=np.roll(logE, -int(naverage/2), axis=0)    
    wvad=logE>trh
    return wvad

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
    print("welcome adrian")
elif(decision==2):
    print("welcome cañarte")
elif(decision==3):
    print("welcome juanjose")
plt.show()
