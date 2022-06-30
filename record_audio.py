from tkinter import Tk,Label,Button,filedialog,Entry,StringVar,messagebox
import glob
import pyaudio
import os
import wave
import threading

class Audio_Creation:
    def __init__(self,tipo):
        self.ventana = Tk()
        self.ventana.title('Grabadora de Audio mp3 sobre la palabra "{}"'.format(tipo))
        self.tipo = tipo
        #VARIABLES INICIALES
        self.directorio_actual=StringVar()
        self.grabando=False
        self.reproduciendo=False
        self.CHUNK=1024
        self.data=""
        self.stream=""
        self.audio=pyaudio.PyAudio() 
        self.f=""

        #CONTADOR DE TIEMPO
        self.time = Label(self.ventana, fg='green', width=20, text="0:00:00", bg="black", font=("","30"))
        self.time.place(x=10,y=20)
        self.ventana.geometry("488x97")

        #BOTONES 
        self.btnIniciar=Button(self.ventana, fg='blue',width=16, text='Grabar', command=self.iniciar)
        self.btnIniciar.place(x=65,y=71)
        self.btnParar=Button(self.ventana, fg='blue', width=16, text='Parar', command=self.parar)
        self.btnParar.place(x=187,y=71)
        self.btnAbrir=Button(self.ventana, text="Abrir",width=16,command=self.abrir)
        self.btnAbrir.place(x=309,y=71)

        self.etDir=Entry(self.ventana,width=77,bg="lavender",textvariable=self.directorio_actual)
        self.etDir.place(x=10,y=0)

        self.dire()

    def clear_contador(self):
        global contador,contador1,contador2
        contador=0
        contador1=0
        contador2=0

    def dire(self):
        self.directorio_actual.set(os.getcwd())

    def iniciar(self):
        global grabando
        global proceso
        global act_proceso
        self.clear_contador()
        audio=pyaudio.PyAudio()
        self.bloqueo('disabled')
        grabando=True
        FORMAT=pyaudio.paInt16
        CHANNELS=1
        RATE=44100
        act_proceso=True
        archivo="dataset/{}/grabacion.wav".format(self.tipo)
        t1=threading.Thread(target=self.grabacion, args=(FORMAT,CHANNELS,RATE,self.CHUNK,audio,archivo))
        t=threading.Thread(target=self.cuenta)
        t1.start()
        t.start()

    def formato(self,c):
        if c<10:
            c="0"+str(c)
        return c
        
    def cuenta(self):
        global proceso
        global contador,contador1,contador2
        self.time['text'] = str(contador1)+":"+str(self.formato(contador2))+":"+str(self.formato(contador))
        contador+=1
        if contador==60:
            contador=0
            contador2+=1
        if contador2==60:
            contador2=0
            contador1+=1
        proceso=self.time.after(1000, self.cuenta)

    def abrir(self):
        global data
        global stream
        global f
        global reproduciendo
        self.clear_contador()
        audio=pyaudio.PyAudio()
        open_archive=filedialog.askopenfilename(initialdir = "dataset/{}/".format(self.tipo),
                    title = "Seleccione archivo",filetypes = (("wav files","*.wav"),
                    ("all files","*.*")))
        if open_archive!="":
            try:
                reproduciendo=True
                f = wave.open(open_archive,"rb")
                stream = audio.open(format = audio.get_format_from_width(f.getsampwidth()),  
                            channels = f.getnchannels(),  
                            rate = f.getframerate(),
                            output = True)
                data = f.readframes(self.CHUNK)
                self.bloqueo('disabled')
                t=threading.Thread(target=self.cuenta)
                t.start()
                t2=threading.Thread(target=self.reproduce)
                t2.start()
            except:
                messagebox.showwarning("ERROR","No se pudo abrir al archivo especificado")
                reproduciendo=False

    def reproduce(self):
        global data
        global stream
        global f
        
        while data and reproduciendo==True:  
            stream.write(data)  
            data = f.readframes(self.CHUNK)  
    
        stream.stop_stream()  
        stream.close()  
    
        self.audio.terminate()
        self.time.after_cancel(proceso)
        #print("FIN")
        self.bloqueo('normal')

    def bloqueo(self,s):
        self.btnIniciar.config(state=s)
        self.btnAbrir.config(state=s)
        
    def parar(self):
        global grabando
        global reproduciendo
        if grabando==True:
            grabando=False
            self.time.after_cancel(proceso)
            self.clear_contador()
        elif reproduciendo==True:
            reproduciendo=False
        self.bloqueo('normal')

    def direc(self):
        directorio=filedialog.askdirectory()
        if directorio!="":
            os.chdir(directorio)
            self.dire()

    def grabacion(self,FORMAT,CHANNELS,RATE,CHUNK,audio,archivo):
        
        stream=audio.open(format=FORMAT,channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)

        frames=[]

        #print("GRABANDO")
        while grabando==True:
            data=stream.read(CHUNK)
            frames.append(data)
        #print("fin")

        #DETENEMOS GRABACIÓN
        stream.stop_stream()
        stream.close()
        audio.terminate()

        grabs = glob.glob('dataset/{}/*.wav'.format(self.tipo))

        count=0
        for i in grabs:
            if "grabacion" in i:
                count+=1
        if count>0:
            archivo="dataset/"+self.tipo+"/grabacion"+"("+str(count)+")"+".wav"
            
        waveFile = wave.open(archivo, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
            

#CREAR VENTANA

 
    def iniciar_programa(self):
        self.ventana.mainloop()