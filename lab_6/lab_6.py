import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import time


# scipy.signal.freqz(b,a)


def ex_1(lista_B = [0.01,5,100] ):

    frecvente = [1,1.5,2,4]
    B = 1
    timp = np.linspace(-3,3,1000)
    x = lambda t : np.sinc(t*B)**2

    x_caciula = lambda t,timp_n,x_n: np.dot(x_n,
                                 np.sinc(
                                     (t -
                                      timp_n.reshape(-1,1)
                                      )/Ts
                                      ))

    for B in lista_B:

        fig , axs = plt.subplots(len(frecvente)+1)

        plt.subplots_adjust(hspace = 0.7)

        fig.suptitle(f"B = {B}")

        axs[0].plot(timp,x(timp))
        axs[0].set_title("Functia originala")


        for i,frec in  enumerate(frecvente):

            Ts = 1/frec


            n_min = np.ceil(-3 / Ts)
            n_max = np.floor( 3 / Ts)
            
            timp_esantioane = np.arange(n_min,n_max+1) *Ts

            axs[i+1].plot(timp,x(timp))
            axs[i+1].stem(timp_esantioane,x(timp_esantioane))
            axs[i+1].plot(timp , x_caciula(timp,timp_esantioane,x(timp_esantioane)), linestyle = "dotted")
            axs[i+1].set_title(f"frecv = {frec}")

        plt.savefig(f"imagini/ex_1/ex_1_B = {B:g}.pdf")
        plt.show()

def ex_2(N = 100,A = 25):
    x = np.random.rand(N)
    dreptunghi = np.zeros(N)
    dreptunghi[A:N-A] = 1 # centram dreptunghiul in mijloc

    #    ceva = lambda x,y: [(axs[i][y].plot(x) , x:=np.convolve(x,x)) for i in range(4)]

    fig, axs = plt.subplots(4,2)
    plt.subplots_adjust(hspace = 0.7 )
    for i in range(4):
        axs[i][0].plot(x)
        x = np.convolve(x,x)
    
    for i in range(4):
        axs[i][1].plot(dreptunghi)
        dreptunghi = np.convolve(dreptunghi,dreptunghi)

    #Observam cu dreptunghiul se face triunghi si dupa se ronujeste, si numerele random
    # ajung sa se rounjeasca si sa nu mai para random, deja dupa prima operatie se vede
    # un fel de triunghi, functia ajunge sa fie crescatoare pana la mijloc si dupa
    # descrescatoare

    plt.savefig("imagini/ex_2.pdf")
    plt.show()
    

def ex_3(N = 9999):
    p = np.random.randint(999999999999999,size=N+1)
    q = np.random.randint(999999999999999,size=N+1)

    timp_start = time.time()

    x = np.convolve(p,q)

    timp_final = time.time() 

    timp_convolutie =  timp_final - timp_start

    timp_start = time.time()
    len_rezultat  = len(p) + len(q) + 1

    p_fft = np.fft.fft(p, len_rezultat)
    q_fft = np.fft.fft(q, len_rezultat)

    r_fft = p_fft * q_fft

    r = np.round(np.real(np.fft.ifft(r_fft))).astype(int)
    timp_final = time.time()

    print(f"Timp prin convolutie:{timp_convolutie}\nTimpul prin fft:{timp_final- timp_start}")

    #Timp prin convolutie:0.031327247619628906
    #Timpul prin fft:0.0036077499389648438

    #Timp prin convolutie:0.03269672393798828
    #Timpul prin fft:0.0038433074951171875

    #Timp prin convolutie:0.03269672393798828
    #Timpul prin fft:0.0038433074951171875

    # Se observa cum prin fft este considerabil mai rapid


def ex_4(n = 20, d = 6):

    x = np.random.uniform(-10,10,n)

    y = np.roll(x,d)

    timp_start_1 = time.time()
    formula_1 = lambda x,y: np.fft.ifft(np.conj(np.fft.fft(x))*np.fft.fft(y))
    rez_1 = formula_1(x,y)
    timp_final_1 = time.time()

    timp_start_2 = time.time()
    formula_2 = lambda x,y: np.fft.ifft(np.fft.fft(y)/np.fft.fft(x))
    rez_2 = formula_2(x,y)
    timp_final_2 = time.time()

    print(x)
    print(y)
    print(np.round(rez_1.real))


    print(f"formula1: {timp_final_1-timp_start_1}formula 1 = {np.argmax(rez_1)}\nin {timp_final_2 - timp_start_2}formula 2 = {np.argmax(rez_2)}")
    # Folosind prima formula, obtinem 'd' daca ne uitam la indexul celui mai mare numar.
    # Folosind a doua formula la indexul 'd' gasim 1 iar 0 in rest (ca un fel de dirac, 1 undeva si 0 in rest ) 
    # se observa si cum formula 2 este considerabil mai rapida


def ex_5(Nw=200, f = 100, Fs = 2000):

    N_total = 2*Nw

    durata_totala = N_total/ Fs

    timp = np.linspace(0,durata_totala,N_total)

    x = lambda t: np.sin(2*f*np.pi*t)

    dreptunghi = lambda N: np.ones(N)
    hanning = lambda N: 0.5 - 0.5*np.cos(2*np.pi*np.arange(N)/(N))

    zerouri = np.zeros(int(Nw/2))

    w_dreptunghi = np.concatenate([zerouri,dreptunghi(Nw),zerouri])
    w_hanning = np.concatenate([zerouri,hanning(Nw),zerouri])
    

    fig, axs = plt.subplots(4,2)
    plt.subplots_adjust(hspace=1)


    axs[0][0].plot(timp, x(timp))
    axs[0][0].set_title('Semnal original')

    axs[0][1].axis('off')

    axs[1][0].plot(timp, x(timp)*w_dreptunghi)
    axs[1][0].set_title('Semnal * dreptunghi')
    
    axs[1][1].plot(timp,w_dreptunghi)
    axs[1][1].set_title('dreptunghi')


    axs[2][0].plot(timp, x(timp)*w_hanning)
    axs[2][0].set_title('Semnal * hanning')

    axs[2][1].plot(timp,w_hanning)
    axs[2][1].set_title('hanning')

    axs[3][0].plot(timp,x(timp)*w_dreptunghi*w_hanning)
    axs[3][0].set_title('semnal * dreptunghi * hanning')
    
    axs[3][1].plot(timp,w_hanning*w_dreptunghi)
    axs[3][1].set_title('dreptunghi * hanning')
    plt.savefig("imagini/ex_5.pdf")
    plt.show()

def main():
    ex_2()
    ex_5()
main()