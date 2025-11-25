import math as math
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os
import time
import scipy
from pprint import pprint

def is_power(n):
    """#credit: https://www.geeksforgeeks.org/python/python-program-to-find-whether-a-no-is-power-of-two/
    2^x   (binar) = 100000...
    2^x-1 (binar) = 011111...
                            &
                    000000
    and - scurtcircuit
    """
    return n > 0 and (n & (n -1 )) == 0


def fft(x):
    """
    #credits tutorial folosit
    https://brianmcfee.net/dstbook-site/content/ch08-fft/FFT.html
    """
    N = len(x)

    if N == 1:
        return x
    

    X_par = fft(x[0::2])
    X_impar = fft(x[1::2])

    X = np.zeros(N,dtype=complex)

    for m in range(N):
        m_alias = m % (N//2)
        X[m] = X_par[m_alias] + np.exp(-2j * np.pi * m / N) * X_impar[m_alias]

    return X


# def xfft(x):
#     N = len(x)
#     if not is_power(N):
#         N = np.power(2,int(np.floor(np.log2(N)+1)),dtype=int)
#     x = np.pad(x,(0, N-len(x)), mode ="constant" )
#     return fft(x)

# iterare prin np.array:
# for i in np.nditer(arr):

def e_putere(n):
    return math.e**(-2*np.pi*1j*n)

def fourier(x):
    x = np.asarray(x , dtype=complex)
    
    N = x.shape[0]

    n = np.arange(N).reshape(N, 1)
    k = np.arange(N).reshape(1, N)
    W = np.exp(-2j * np.pi * n @ k / N)
    return W @ x



# def fourier(N):

#     matrice = []
#     for v in range(N):
#         matrice.append([])
#         for w in range(N):
#             matrice[v].append(e_putere(w*v/N))
#     return matrice

# def arr(N=4):
#     A = np.linspace(0,N-1,N)
#     A = np.reshape(A,(N,1))
#     B = np.transpose(A)
#     AB = A @ B
#     print(A)
#     print("~~"*40)
#     print(B)
#     print("~~"*40)
#     print(AB)
#     # x = np.ones((N,N), dtype= np.complex64)
#     # x = x*np.exp(np.e,(-2*np.pi*1j*N))

def time_of(func: callable ,arg):
    t_0 = time.perf_counter()
    func(arg)
    t_1 = time.perf_counter()
    
    return t_1 - t_0 


def bench():
    list_N = [128,256,512,1024,2048,4096,8192]
    # nu merge cu 16384, am incercat, procesul a fost omorat


    # fourier()
    # fft()
    # np.fft.fft()

    list_fou = []
    list_fft = []
    list_np = []

    for N in list_N:
        timpspace = np.linspace(0,10,N)
        X_d = np.sin(40*np.pi*timpspace)
        
        list_fou.append(time_of(fourier,X_d))   
        list_fft.append(time_of(fft,X_d))
        list_np.append(time_of(np.fft.fft,X_d))

    return list_N, list_fou , list_fft, list_np

def plot_log(list_N, list_fou, list_fft, list_np):
    plt.plot(list_N, list_fou, '-o', label='DFT cu matrice')
    plt.plot(list_N, list_fft, '-o', label='FFT al meu')
    plt.plot(list_N, list_np,  '-o', label='Numpy')
    plt.yscale('log')
    plt.xticks(list_N, list_N)
    plt.ylabel('Timp')
    plt.xlabel('N')
    plt.legend()
    plt.savefig("imagini/timpi.pdf",format = "pdf")

def ex_1():
    plot_log(*bench())

def ex_2():
    timp_gresit = np.linspace(0,3,6,endpoint=False) # 2hz (limita  Nyquist)
    timp = np.linspace(0,3,1000)

    x = lambda t: np.sin(2*np.pi*t) # 1hz
    y = lambda t: np.sin(4*np.pi*t) # 2 Hz
    z = lambda t: np.sin(6*np.pi*t) # 3 Hz

    fig , axs = plt.subplots(4)

    for ax in axs.flatten():
        ax.set_ylim(-1,1)

    axs[0].plot(timp,x(timp))

    axs[1].plot(timp,x(timp))
    axs[1].plot(timp_gresit,x(timp_gresit),'oy')

    axs[2].plot(timp,z(timp))
    axs[2].plot(timp_gresit,z(timp_gresit),'oy')

    axs[3].plot(timp,y(timp))
    axs[3].plot(timp_gresit,y(timp_gresit),'oy')

    plt.savefig("imagini/ex_2.pdf", format = "pdf")
    plt.show()

def ex_3():
    timp_gresit = np.linspace(0,3,6) # 2hz (limita  Nyquist+1)
    timp = np.linspace(0,3,1000)

    x = lambda t: np.sin(2*np.pi*t) # 1hz
    y = lambda t: np.sin(4*np.pi*t) # 2 Hz
    z = lambda t: np.sin(6*np.pi*t) # 3 Hz

    fig , axs = plt.subplots(4)

    for ax in axs.flatten():
        ax.set_ylim(-1,1)

    axs[0].plot(timp,x(timp))

    axs[1].plot(timp,x(timp))
    axs[1].plot(timp_gresit,x(timp_gresit),'oy')

    axs[2].plot(timp,z(timp))
    axs[2].plot(timp_gresit,z(timp_gresit),'oy')

    axs[3].plot(timp,y(timp))
    axs[3].plot(timp_gresit,y(timp_gresit),'oy')

    plt.savefig("imagini/ex_3.pdf", format = "pdf")
    plt.show()

def ex_6():
    rate , vocale = scipy.io.wavfile.read("audio/vocale.wav")
    
    N = len(vocale)

    marime_grup = int(0.01*N)

    suprapunere = int(marime_grup/2)

    ferestre = [vocale[i:i+marime_grup] for i in range(0, N-marime_grup +1, suprapunere)]


    fft_lista = np.fft.fft(ferestre)

    fft_absolut = np.abs(fft_lista)

    spectrograma , nimic = np.split(fft_absolut, 2, axis =1)
    #pastram doar jumatate, fiind numere reale


    spectrograma = spectrograma.T



    spectrograma_db = 20 * np.log10(spectrograma + 1e-10)

    plt.figure(figsize=(10, 6))
    
    plt.imshow(spectrograma_db, 
               aspect='auto', 
               origin='lower', 
               cmap='jet')

    plt.title("Spectograma")
    plt.ylabel("Frecventa")
    plt.xlabel("Timp")
    plt.colorbar(label="Amplitudine (dBFS)")    
    
    plt.savefig("imagini/ex_6.pdf", format = "pdf")

    plt.show()


def main():
    ex_1()
    ex_2()
    ex_3()
    ex_6()

main()
