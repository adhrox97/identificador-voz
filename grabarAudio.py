import pyaudio
import wave

#DEFINIMOS PARAMETROS
FORMAT=pyaudio.paInt16
CHANNELS=1
RATE=16000
CHUNK=1024
duracion=4
archivo="Datos/juanjose/juanjose40.wav"

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
