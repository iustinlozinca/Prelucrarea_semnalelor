import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime
from scipy.signal import butter , cheby1, filtfilt


def citire(csv='data/Train.csv'):
    dtype=[(''),(),()]
    x = np.genfromtxt(csv,dtype=float, delimiter=',',skip_header=1, usecols=(2))
    return x


def ex_b(x,
         lungime = 3*24,
         windows = [5, 9, 13, 17]):
    plt.figure(figsize=(12, 6))
    plt.plot(range(lungime), x, label="Semnal original")
    

    for w in windows:
        x_neted = np.convolve(x, np.ones(w), 'valid')/w
        
        # Centram
        offset = (lungime - len(x_neted)) // 2
        plt.plot(range(offset, offset + len(x_neted)), x_neted, label=f'w={w}')

    plt.title("Ex 6 (b)")
    plt.xlabel("Ore")
    plt.ylabel("Nr. masini")
    plt.legend()
    plt.grid(True)
    plt.savefig("imagini/ex_6/ex_b.pdf")
    plt.show()



def ex_c_d_e(x,
             prag_taiere =12,
             ordin = 5,
             rp_lista = [
                        # 0.1,
                        # 1,
                        # 3,
                         5
                         #10
                         #30
                         # Dupa mai multe incercari, indiferent de ordin, rp mai mare de 10
                         # ajunge sa fie prea deformat si nu cred ca este util la ceva
                         # 
                             ]
                             ):

    Fs = 1 / 3600 # Hz
    F_nyquist = Fs / 2 # Hz
        
         
    frecventa_taiere = 1 / (prag_taiere * 3600) # Hz
        
    Wn =frecventa_taiere / F_nyquist
        
    for rp in rp_lista:
        b_butter , a_butter = butter(ordin ,Wn, btype = 'low')
        
        b_cheb , a_cheb = cheby1(ordin, rp , Wn , btype = 'low')
        
        y_butter = filtfilt(b_butter, a_butter, x)
        y_cheb = filtfilt(b_cheb, a_cheb, x)

        plt.figure(figsize=(18, 6))
        plt.plot(x, 'k-', label='Original (Raw)')
        plt.plot(y_butter, 'b-', label=f'Butterworth')
        plt.plot(y_cheb, 'r--', label=f'Chebyhshev \nrp={rp}dB\nordin={ordin}')
        plt.xlabel('Ora')
        plt.ylabel('Numar masini')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"imagini/ex_6/ex_c_d_e{ordin:g}.pdf")
        plt.show()


def ex_f(x,ordini = [#2,
                     5,
                     9,
                     15] ):
    # Graficul cu Butter arata asemanator la majoritatea ordinelor iar cel cu 
    # Chebyhshev la ordin 20 devine inutilizabil si se duce mult in negativ
    # iar la ordin 2 cand rp creste Chebyhshev se indeparteaza foarte mult de
    # graficul original, ar trebui sa fac +2 sa il aduc la loc

    for ordin in ordini:
        ex_c_d_e(x,ordin= ordin)

    # Pentru exercitiul f eu as alege filtrul Butterworth cu care spre exemplu as putea
    # sa identific perioade cu sau fara traficavand in vedere ca graficul esantioanelor
    # este discret si "tepos" de asemenea cu un filtru Butterworth se poate intreba in orice moment
    # "cam cat trafic este" fara sa fiu limitat la ore exacte 

def main(start_index = 105, lungime = 3*24):

    # (a)
    x_full = citire()
    x = x_full[start_index : start_index + lungime]
    
    ex_f(x)

main()