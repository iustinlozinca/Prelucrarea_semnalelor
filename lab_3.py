from pprint import pprint
import math as math
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as anim
pi = np.pi

#exemplu sintaxa math.e**(1j*5)
# semnal discret x[n]

def e_putere(n):
    return math.e**(-2*pi*1j*n)

# artefact
# def fourier(func,N):
#     lista = []
#     for n in range(N):
#         def aux[n](omega):
#             return (func(n)*e_putere(n,omega,N))
#         lista.append(aux)
#     return lista

def fourier(N):
    matrice = []
    for v in range(N):
        matrice.append([])
        for w in range(N):
            matrice[v].append(e_putere(w*v/N))
    return matrice

def ex_1(N):

    F = fourier(N)
    F = np.array(F)
    FH = np.transpose(F)
    FH = np.conj(FH)
    
    Identitate = np.eye(N)
    pprint(F)
    print('=='*40)
    pprint(FH)
    print("~~"*40)
    pprint(np.round(FH @ F))
    pprint(Identitate)
    print(np.allclose(FH @ F,N*Identitate))

    fig , axs = plt.subplots(N)

    i = 0
    for linie in F:
            real = np.real(linie)
            imag = np.imag(linie)
            axs[i].plot(real)
            axs[i].plot(imag)
            i+=1
    os.makedirs("imagini_3", exist_ok=True)
    plt.savefig("imagini_3/ex_1.pdf", format = 'pdf')

def ex_2():
    timp = np.linspace(0,100,50)
    x = np.sin(2 +np.pi*2*100*timp)
    y = x*np.exp(-1j*np.pi*timp)
    fig , axs = plt.subplots(1,2)
    axs[0].plot(timp,x)
    axs[1].plot(y.real,y.imag)
    os.makedirs("imagini_3", exist_ok=True)
    plt.savefig("imagini_3/ex_2.pdf", format="pdf")
    for elem in y:
        axs[1].stem(y.real, y.imag)
        plt.show()

         




def main():

    ex_2()


main()