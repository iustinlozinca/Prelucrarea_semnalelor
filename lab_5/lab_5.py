import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime


#componenta continua = prima componenta din fft



# ex_1_a 
# (a)
# train.csv 18288 esantioane din ora in ora
# frecventa 1/ora sau 1/60 pe minut sau 1/3600 pe secunda, 1/360[Hz] frecventa esantionare


def ex_1_b():
    # - 25-09-2014 23:00 - 25-08-2012 00:00 

    data_inceput = "25-09-2014 23:00"
    data_final = "25-08-2012 00:00"
    fmt = "%d-%m-%Y %H:%M"
    t1 = datetime.datetime.strptime(data_inceput,fmt)
    t2 = datetime.datetime.strptime(data_final,fmt)

    delta = t1-t2
    print(str(delta.days/365)+" ani sau")
    print(str(delta.days)+" zile sau")
    print(str(delta.seconds)+" secunde")

# ex_1_c (c)
# Considerand ca semnalul este esantionat optim conform Nyquist
# => Frecventa max *2 < frecv esantionare
# => frecventa max ~ 1/2*1/3600
# => freventa max ~ 1/7200
# (folosesc ~ pentru ca frecventa maxima nu poate fii 1/7200 pentru ca
# semnul este <, nu <=)


def citire(csv='data/Train.csv'):
    dtype=[(''),(),()]
    x = np.genfromtxt(csv,dtype=float, delimiter=',',skip_header=1, usecols=(2))
    return x

def semnal():
    x = citire()
    N = len(x)
    Fs = 1/3600
    X = np.fft.fft(x)
    X_modul= np.abs(X / N)

    X_modul , nimic= np.split(X_modul, 2, axis= 0)

    f = Fs*np.linspace(0,N/2,int(N/2))/N

    return x,X,X_modul,N,Fs,f


def ex_1_d():
    x,X,X_modul,N,Fs,f = semnal()


    plt.plot(f,X_modul)
    plt.savefig("imagini/ex_1_d.pdf")
    plt.show()


def ex_1_e(plot = True):
    #Din graficul de la exercitiul anterior se observa un varf la inceputul graficului
    #media semnalului fiind diferita de 0 inseamna ca aceea este componenta continua
    x,X,X_modul,N,Fs,f = semnal()

    x = x - np.mean(x)

    X = np.fft.fft(x)
    X_modul= np.abs(X / N)

    X_modul , nimic= np.split(X_modul, 2, axis= 0)

    f = Fs*np.linspace(0,N/2,int(N/2))/N
    if plot == True:
        plt.plot(f,X_modul)
        plt.savefig("imagini/ex_1_e.pdf")
        plt.show()

    return x,X,X_modul,N,Fs,f


def ex_1_f():
    x,X,X_modul,N,Fs,f = ex_1_e(False)
    
    top_4 = np.argsort(X_modul)[-4:][::-1]

    print("hz","~~"*40)
    for i in top_4:
        print(X_modul[i],f": frecventa in hz {f[i]}")
    print("ore","~~"*40)
    #convertim in ore
    for i in top_4:
        print(X_modul[i],f": frecventa in ore {(1/f[i])/(3600)}")
    print("zile","~~"*40)
    #convertim in zile
    for i in top_4:
        print(X_modul[i],f": frecventa in zile {(1/f[i])/(3600*24)}")

    #interpretarea rezultatelor:
    # Cea mai evidenta  este freceventa
    #  de pe locul 3 (0.99999 in zile)

    #locul 1 (761.91666 in zile) reprezinta faptul ca numarul masinilor a crescut
    # pe parcursul celor 2 ani, foarte apropiat de intreaga perioada (761 zile)

    # locul 2 (380.95 in zile) este aproximativ un an,
    # eu cred ca nu este mai apropiat
    # de un an pentru ca numarul masinilor a crescut 

    # locul 4 (253.97)nu stiu exact ce reprezinta dar este
    # cam o treime din zilele totale

def ex_1_g():
    #9-12-2013 # 11304
    #12071,09-01-2014 23:00,210
    x = citire()

    index_inceput = 11304 #9-12-2013
    esantioane = 30 * 24
    index_final = index_inceput + esantioane


    x = x[index_inceput : index_final]
    plt.plot(x)
    plt.savefig("imagini/ex_1_g.pdf")
    plt.show()
def low_pass(x , vector_frecventa, prag):
    masca = np.abs(vector_frecventa) <= prag
    return x*masca


def ex_1_i(prag_ore = 12):
    x,X,X_modul,N,Fs,f = semnal()

    vector_frecventa = np.fft.fftfreq(N ,d = 1/Fs)


    prag = 1 / (prag_ore* 3600)

    x_nou = low_pass(X,vector_frecventa, prag)

    x_nou = np.real(np.fft.ifft(x_nou))

    fig , axs = plt.subplots(2)

    axs[0].plot(x)
    axs[0].set_title("vechi")

    axs[1].plot(x_nou)
    axs[1].set_title("Nou")
    plt.savefig("imagini/ex_1_i.pdf")
    plt.show()

    # Exercitiul I
    # Am ales sa elimin frecventele cu o perioada mai mica de 10 ore
    # pentru a netezi graficul si pentru a se observa mai usor
    # tendintele generale ale traficului


#Exercitiul H

# Stiim ca esantiaonele sunt din ora in ora
# Stiim ca de la inceput pana la final trec putin peste 2 ani

# Saptamanile si zilele sunt foarte vizibile pe grafic in special dupa fourier (ca la
# exercitiul 1 f), weekend-ul este usor
# observabil, folosind aceasta informatie putem sa numaram zilele sa vedem in ce zi
# a saptamanii a inceput esantionarea

# Putem presupune ca anumite ore sunt mai aglomerate in zilele lucratoare (ex: cand lumea
# pleaca de la servici sau cand se duce la servici) folosind aceasta informatie
# putem incerca sa aflam ora de la care incepe esantionarea

# Suprapunem primul an cu al doilea an din semnal si incercam sa ghicim unde ar fii 
# sarbatori. Exemplu: Intre craciun si revelion este o perioada scurta deci poate
# daca suprapunem graficele observam un tipar. Daca suntem destul de siguri pe
# ce am ghicit din grafic putem numara zilele pana la o urmatoare sarbatoare
# sau la una de dinainte sa vedem daca observam un tipar similar.
# Presupun ca in zilele libere nu ar mai fii atat de mult trafic la orele
# de plecare la servici dar ar fii mai mult trafic pe parcursul zilei de munca

# Daca putem afla unde in esantioane este o sarbatoare ne putem folosi sa aflam
# anul in care a fost facuta esantionarea. Ne uitam in ce zi a saptamanii pica
# sarbatoarea pe care o gasim (ex: identificam craciunul intr-o zi de marti)
# si dupa verificam toti anii din prezent pana in 1880 (primul search pe google
# spune ca pana in 1880 nu existau masini) pentru ani in care craciunul a picat
# intr-o zi de marti. 1880 este o exagerare pentru ca nu erau destule masini
# cat pentru un esantion sa ajunga la 100.

# Neajunsuri:
# Ar putea unul dintre ani sa fie bisecti si atunci calculul legat de sarbatori
# poate nu mai functioneaza.
# Presupunerile legate de sarbatori ar putea fi gresite
# Presupunerile facute ar putea fi gresite

def main():
    ex_1_d()
    ex_1_e()
    ex_1_g()
    ex_1_i()

main()