import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile 
import scipy.signal  
import sounddevice 

def lab_1_ex_2():

    def sinus(t,frecventa):
        return np.sin(2*np.pi*t*frecventa)
    timp = []
    fig, axs = plt.subplots(5)
    plt.subplots_adjust(hspace=1.5)
    # (a)
    i = 0
    axs[i].set_title("400hz , 1600 esantioane")
    timp.append(np.linspace(0, 0.01, 1600, endpoint= False))
    axs[i].plot(timp[i],a := sinus(timp[i],400))
    
    # (b)
    i = 1
    axs[i].set_title("800hz, 1600 esantioane") #graficul arata urat doar ca daca esantionez mai putin practic nu e acelasi grafic
    timp.append(np.linspace(0, 3, 4801,endpoint= False))  
    axs[i].set_ylim(-1,1)
    axs[i].plot(timp[i],b := sinus(timp[i],800))
    # frecventa = 800 / s
    # 3 secunde => esantioane/3 = esantioane per secunda
    # => 2*800*3= 4800 esantioane /3s
    # Intrebare: de ce valori dupa 4800 arata grafice diferite
    # si de ce 4800 nu arata un grafic


    # (c)
    i = 2
    axs[i].set_title("c")
    timp.append(np.linspace(0, 5, 170, endpoint= False)  )
    axs[i].set_ylim(-1,1)
    axs[i].plot(timp[i],c := np.floor(sinus(timp[i],240)))

    # (d)
    i = 3
    axs[i].set_title("d")
    timp.append(np.linspace(0, 4, 150))  
    axs[i].plot(timp[i],d := np.sign(sinus(timp[i],300)))


    # (e)

    # i = 4
    # axs[i].set_title("e")
    # arr = np.random.rand(x,y)
    # timp.append(np.linspace())
    plt.close()
    return a,b,c,d




def ex_1():
    sinus = lambda t: np.sin(6*np.pi * t)
    cosinus = lambda t: np.cos(6*np.pi*t-np.pi/2)

    timp = np.linspace(0,1,500)


    fig, axs = plt.subplots(3)
    plt.subplots_adjust(hspace=0.7)


    axs[0].plot(timp,sinus(timp), 'y')
    axs[0].plot(timp,cosinus(timp), 'r')
    axs[0].set_title('cosinus si sinus')

    axs[1].plot(timp,sinus(timp), 'y')
    axs[1].set_title('sinus')

    axs[2].set_title('cosinus')
    axs[2].plot(timp,cosinus(timp), 'r')

    plt.savefig("imagini/ex_1.pdf", format = 'pdf')
    plt.show()


def ex_2_a():

    timp = np.linspace(0,1,500)
    sinus = []

    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi/6))
    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi/3))
    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi/2))
    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi))



    fig, axs = plt.subplots(5)
    plt.subplots_adjust(hspace=0.7)


    for i in range(4):
        axs[i].plot(timp,sinus[i](timp))
        axs[4].plot(timp,sinus[i](timp))

    plt.savefig("imagini/ex_2_a.pdf", format = "pdf")
    plt.show()




def ex_2_b():
    
    timp = np.linspace(0,1,500)
    sinus = []

    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi/6))
    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi/3))
    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi/2))
    sinus.append(lambda t: np.sin(2*np.pi*t+np.pi))


    x_d = sinus[0](timp) #x discret
    x_norm = np.linalg.norm(x_d)

    z = np.random.normal(0, 1, size = len(timp))
    z_norm = np.linalg.norm(z)

    snr_list = [0.1,1,10,100]

    fig, axs = plt.subplots(4)
    plt.subplots_adjust(hspace=1)
    for i , snr in enumerate(snr_list):
        gamma = x_norm /(np.sqrt(snr)*z_norm)
        x_zgomot = x_d + gamma*z

        axs[i].plot(timp,x_d,'p')
        axs[i].plot(timp,x_zgomot,'r')
        axs[i].set_title(f"snr: {snr}")


    plt.savefig("imagini/ex_2_b.pdf", format = "pdf")
    plt.show()

def ex_3():
    a,b,c,d = lab_1_ex_2()

    rate = int(10e5)
    scipy.io.wavfile.write("sunete/semnal_a.wav",rate,a)

    fs = 44100
    sounddevice.play(a, fs)
    print(a)
    rate, x = scipy.io.wavfile.read("sunete/semnal_a.wav")
    print(x)
    print(x == a)


def ex_4():
    timp = np.linspace(0,5,500)
    
    sinus = lambda t: np.sin(6*np.pi*t+np.pi)
    sawtooth = lambda t: (t%2)/2 -1

    amandoua = lambda t: sinus(t)+sawtooth(t)

    fig, axs = plt.subplots(3)
    plt.subplots_adjust(hspace= 1)

    
    axs[0].plot(timp,sinus(timp))
    axs[0].set_title("sinus")
    axs[0].set_ylim(-2,2)


    axs[1].plot(timp,sawtooth(timp))
    axs[1].set_title("sawtooth")
    axs[1].set_ylim(-2,2)
        
    axs[2].plot(timp,amandoua(timp))
    axs[2].set_title("amandoua")
    axs[2].set_ylim(-2,2) 


    plt.savefig("imagini/ex_4.pdf",format = "pdf")
    plt.show()


def ex_5():

    x = lambda t: np.sin(500*np.pi*t)
    y = lambda t: np.sin(300*np.pi*t)

    fs = 44100

    timp = np.linspace(0,5,fs*5)

    x_d = x(timp)
    y_d = y(timp)

    semnal_concatenat = np.concatenate((x_d,y_d))

    sounddevice.play(semnal_concatenat,fs)
    sounddevice.wait()
    scipy.io.wavfile.write("sunete/ex_5.wav",fs,semnal_concatenat)

    #Observ cum se aude diferenta dintre cele doua, semnalul
    # cu frecventa mai inalta se aude mai "pitigaiat"

def ex_6():
    fs = 500
    a = lambda t: np.sin(t*2*np.pi*  (fs/2)  ) 
    b = lambda t: np.sin(t*2*np.pi*  (fs/4)  )
    c = lambda t: np.sin(t*2*np.pi*  (0/2)  )

    timp = np.linspace(0,1,fs, endpoint= "false")

    fig, axs = plt.subplots(3)

    plt.subplots_adjust(hspace = 1)

    axs[0].plot(timp,a(timp))
    axs[0].set_title("fs/2")

    axs[1].plot(timp,b(timp))
    axs[1].set_title("fs/4")

    axs[2].plot(timp,c(timp))
    axs[2].set_title("0")


    plt.savefig("imagini/ex_6.pdf" , format = "pdf")
    plt.show()


    #Observatie: Cu cat frecventa este mai mare cu atat semnalul este mai "dens" atunci
    # cand il afisam pe grafic, cand avem f = 0 practic semnalul este mereu f(x)=sin(0)
    # de asemenea frecventa de esantionare fiind mica, graficul nu reprezinta in mod
    # cinstit functia initiala

def ex_7():


    x = lambda t: np.sin(300*np.pi*t)
    timp = np.linspace(0,1,1000)

    x_d = x(timp)


    fig, axs = plt.subplots(3)

    plt.subplots_adjust(hspace = 1)

    axs[0].plot(timp,x_d)
    axs[0].set_title("original")

    axs[1].plot(timp[::4],x_d[::4])
    axs[1].set_title("decimat")

    axs[2].plot(timp[1::4],x_d[1::4])
    axs[2].set_title("decimat de la 1")


    #Observatie: Pentru diferitele frecvente incercate am observat
    # cum semnalele arata mai "taios" si cu mai putin detaliu.
    # Daca frecventa functiei este mica atunci esantionarea
    # contine multa redundanta si arata la fel (din ce observ eu)
    # daca frecventa functiei este mai mare atunci dupa decimare
    # se pierd parti mai semnificative din functie si arata lipsit
    # detaliu
    
    plt.savefig("imagini/ex_7.pdf", format = "pdf")
    plt.show()

def ex_8():
    #Inginerii astia si aproximarile lor, urmeaza sa aflu ca pi=3 :)

    timp = np.linspace(-np.pi/2,np.pi/2,10000)

    Pade = lambda a: ((a - (7 * (a**3))/60)
                      /
                      (1+(a**2)/20)
                      )


    zero = lambda t: 0*t

    fig, axs = plt.subplots(5)
    plt.subplots_adjust(hspace= 1)


    axs[0].plot(timp,np.sin(timp))
    axs[0].set_title("sin(x)")
    
    axs[1].plot(timp,timp)
    axs[1].set_title("x")

    axs[2].plot(timp,Pade(timp))
    axs[2].set_title("Pade")

    axs[3].plot(timp,np.sin(timp),'b', label="sin",alpha = 0.5)
    axs[3].plot(timp,timp,'r',label ="sin = x",alpha = 0.5)
    axs[3].plot(timp,Pade(timp),'g',label ="Pade ",alpha = 0.5)
    axs[3].set_title("toate")
    plt.legend()

    axs[4].plot(timp,zero(timp),'b',label="0",alpha = 0.5)
    axs[4].plot(timp,np.sin(timp)-timp,'r',label='err sin(x)=x',alpha = 0.5)
    axs[4].plot(timp,np.sin(timp)-Pade(timp),'g',label='err Pade',alpha = 0.5)
    axs[4].set_title("eroare")
    plt.legend()


    plt.savefig("imagini/ex_8.pdf")
    plt.show()

def ex_8_log():
    timp = np.linspace(-np.pi/2,np.pi/2,10000)

    Pade = lambda a: ((a - (7 * (a**3))/60)
                      /
                      (1+(a**2)/20)
                      )

    zero = lambda t: 0*t

    fig, axs = plt.subplots(5)
    plt.subplots_adjust(hspace= 1)


    axs[0].plot(timp,np.sin(timp))
    axs[0].set_title("sin(x)")
    
    axs[1].plot(timp,timp)
    axs[1].set_title("x")

    axs[2].plot(timp,Pade(timp))
    axs[2].set_title("Pade")

    axs[3].plot(timp,np.sin(timp),'b', label="sin",alpha = 0.5)
    axs[3].plot(timp,timp,'r',label ="sin = x",alpha = 0.5)
    axs[3].plot(timp,Pade(timp),'g',label ="Pade ",alpha = 0.5)
    axs[3].set_title("toate")
    axs[3].legend()

    axs[4].plot(timp,zero(timp),'b',label="0",alpha = 0.5)
    axs[4].plot(timp,np.abs(np.sin(timp)-timp),'r',label='err sin(x)=x',alpha = 0.5)
    axs[4].plot(timp,np.abs(np.sin(timp)-Pade(timp)),'g',label='err Pade',alpha = 0.5)
    axs[4].set_title("eroare")
    axs[4].set_yscale('log')
    axs[4].set_xlim(0)
    axs[4].legend()

    plt.savefig("imagini/ex_8.pdf", format = "pdf")
    plt.show()

def main():
    ex_1()
    ex_2_a()
    ex_2_b()
    # ex_3()
    ex_4()
    # ex_5()
    ex_6()
    ex_7()
    ex_8()
    ex_8_log()

main()