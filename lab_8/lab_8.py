import numpy as np
import matplotlib.pyplot as plt
from time import time

def ex_a(N=1000, plot = True):

    timp = np.arange(N)

    def ecuatie_grad2(a=0.00005,b=0.003,c=2):
        return lambda x: a*x**2 +b*x+c  
    
    def creeaza_sezon(a1 = 5,frec1 = 1/12,a2 = 10,frec2 = 1/50, c=3):
        return lambda x: a1*np.sin(np.pi*frec1*x)+a2*np.sin(np.pi*frec2*x) + c
    
    zgomot = np.random.normal(loc = 0 , scale = 10, size=N)


    trend_functie = ecuatie_grad2()
    sezon_functie = creeaza_sezon()

    trend = trend_functie(timp)
    sezon = sezon_functie(timp)

    y = zgomot + trend + sezon
    if plot == True:
        fig,axs = plt.subplots(4)

        axs[0].plot(timp,y)
        axs[0].set_title("Serie de timp")

        axs[1].plot(timp,trend)
        axs[1].set_title("trend")


        axs[2].plot(timp,sezon)
        axs[2].set_title("sezon")


        axs[3].plot(timp,zgomot)
        axs[3].set_title("zgomot")

        plt.savefig("imagin/ex_a.pdf")
        plt.show()

    else:
        return timp,y,trend,sezon,zgomot

def ex_b(N=1000):
    timp,y,trend,sezon,zgomot = ex_a(N=N,plot = False)
    y_centrat = y - np.mean(y)

    # varianta 1
    autocorelare_correlate = np.correlate(y_centrat , y_centrat, mode = "full")
    autocorelare_correlate = autocorelare_correlate[N-1:]
    autocorelare_correlate_norm = autocorelare_correlate / autocorelare_correlate[0]
    
    # varianta 2
    autocorelare_conv = np.convolve(y_centrat,np.flip(y_centrat), mode = "full")
    autocorelare_conv = autocorelare_conv[N-1:]
    autocorelare_conv_norm = autocorelare_conv / autocorelare_conv[0]

    diferenta = np.sum(np.abs(autocorelare_conv_norm - autocorelare_correlate_norm))
    print(f"diferenta dintre cele doua: {diferenta}")
    decalaje = np.arange(N)
    limita_plot = 200

    plt.plot(decalaje[:limita_plot], autocorelare_conv_norm[:limita_plot], linewidth = 3)
    plt.plot(decalaje[:limita_plot], autocorelare_correlate_norm[:limita_plot],  linewidth = 1)
    plt.grid(True)

    plt.savefig("imagini/ex_b.pdf")
    plt.show()

def ex_c(N= 1000,p = 50):
    timp,y,trend,sezon,zgomot = ex_a(N=N,plot = False)

    media_y = np.mean(y)
    y_centrat = y - media_y


    autocorelare = np.convolve(y_centrat,np.flip(y_centrat), mode = "full")
    autocorelare = autocorelare[N-1:]
    r = autocorelare / autocorelare[0]

    y_p = r[1:p+1]

    gamma_p = np.zeros((p,p))

    for i in range(p):
        for j in range(p):
            gamma_p[i, j] = r[np.abs(i-j)]

    x_star = np.linalg.solve(gamma_p, y_p)

    y_pred = np.zeros(N)

    y_pred[:p] = y[:p]

    for i in range(p , N):

        window = y_centrat[i-p:i]

        window_reversed = np.flip(window)
        
        valoare_prezisa_centrata = x_star @ window_reversed

        y_pred[i] = valoare_prezisa_centrata + media_y

    plt.figure(figsize=(12, 6))
    plt.plot(timp, y, label="Datele originale", color='blue', alpha=0.5)
    plt.plot(timp[p:], y_pred[p:], label=f"Predictie AR, p={p}", color='red', linestyle='--')
    
    plt.title(f"Predictie p={p}")
    plt.legend()
    plt.grid(True)
    plt.savefig("imagini/ex_c.pdf")
    plt.show()




def ruleaza_model(y, p, m):
    N = len(y)
    y_pred = np.zeros(N)
    
    y_pred[:m] = y[:m]
    
    error_sum = 0
    count = 0

    for i in range(m, N):
        
        window_train = y[i-m : i]
        
        media_locala = np.mean(window_train)
        window_train_centrat = window_train - media_locala
        
        autocorelare = np.convolve(window_train_centrat, np.flip(window_train_centrat), mode="full")
        

        autocorelare = autocorelare[m-1:]
        
        if autocorelare[0] == 0:
            r = np.zeros(len(autocorelare))
        else:
            r = autocorelare / autocorelare[0]

        y_p_vec = r[1:p+1]
        
        gamma_p = np.zeros((p, p))
        for r_idx in range(p):
            for c_idx in range(p):
                gamma_p[r_idx, c_idx] = r[np.abs(r_idx - c_idx)]
        
        x_star = np.linalg.solve(gamma_p, y_p_vec)

        istoric_recent = y[i-p : i] - media_locala
        istoric_reversed = np.flip(istoric_recent)
        
        valoare_prezisa = (x_star @ istoric_reversed) + media_locala
        y_pred[i] = valoare_prezisa
        
        error_sum += (y[i] - valoare_prezisa) ** 2
        count += 1
        
    mse = error_sum / count if count > 0 else float('inf')
    return mse, y_pred





def ex_d(N = 1000):
    timp, y, trend, sezon, zgomot = ex_a(N=N, plot=False)

    p_lista = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50,60,70,100]
    m_lista = [10,50, 100, 150, 200, 300]

    best_mse = float('inf')
    best_params = (0, 0)
    best_pred = y
    i = 0
    for p in p_lista:
        for m in m_lista:
            if m <= p:
                continue
            i += 1

            durata_start = time()
            mse, y_pred_curent = ruleaza_model(y, p, m)
            durata = time() - durata_start

            print(f"incercare = {i} ; p = {p} ; m = {m} ; mse = {mse} ; timp = {durata:.2}")
            if mse < best_mse:
                    best_mse = mse
                    best_params = (p, m)
                    best_pred = y_pred_curent

    print(best_params)
    p,m = best_params
    
    plt.figure(figsize=(12, 6))
    plt.plot(timp, y, label="Datele originale", color='blue', alpha=0.5)
    plt.plot(timp[m:], best_pred[m:], label=f"Predictie AR, p={p}, m={m}", color='red', linestyle='--')
    
    plt.title(f"Predictie p={p}")
    plt.legend()
    plt.grid(True)
    plt.savefig("imagini/ex_d.pdf")
    plt.show()


def main():
    ex_d()

main()


# (5, 200)
# incercare = 1 ; p = 3 ; m = 10 ; mse = 156.1449821763444 ; timp = 0.014
# incercare = 2 ; p = 3 ; m = 50 ; mse = 148.1318434393323 ; timp = 0.015
# incercare = 3 ; p = 3 ; m = 100 ; mse = 131.84652926771534 ; timp = 0.015
# incercare = 4 ; p = 3 ; m = 150 ; mse = 131.70178270471564 ; timp = 0.015
# incercare = 5 ; p = 3 ; m = 200 ; mse = 129.74078613362116 ; timp = 0.016
# incercare = 6 ; p = 3 ; m = 300 ; mse = 131.71305626181913 ; timp = 0.017
# incercare = 7 ; p = 5 ; m = 10 ; mse = 159.6614149401554 ; timp = 0.022
# incercare = 8 ; p = 5 ; m = 50 ; mse = 150.1226824448914 ; timp = 0.022
# incercare = 9 ; p = 5 ; m = 100 ; mse = 130.67364127189967 ; timp = 0.022
# incercare = 10 ; p = 5 ; m = 150 ; mse = 130.0085657599477 ; timp = 0.022
# incercare = 11 ; p = 5 ; m = 200 ; mse = 127.74993593643953 ; timp = 0.022
# incercare = 12 ; p = 5 ; m = 300 ; mse = 127.76299179721497 ; timp = 0.022
# incercare = 13 ; p = 7 ; m = 10 ; mse = 156.42264104560203 ; timp = 0.034
# incercare = 14 ; p = 7 ; m = 50 ; mse = 154.30827205757424 ; timp = 0.033
# incercare = 15 ; p = 7 ; m = 100 ; mse = 132.97486002416457 ; timp = 0.033
# incercare = 16 ; p = 7 ; m = 150 ; mse = 131.60782818336932 ; timp = 0.033
# incercare = 17 ; p = 7 ; m = 200 ; mse = 129.24605870802762 ; timp = 0.032
# incercare = 18 ; p = 7 ; m = 300 ; mse = 128.10496555596742 ; timp = 0.031
# incercare = 19 ; p = 10 ; m = 50 ; mse = 159.80984201002704 ; timp = 0.059
# incercare = 20 ; p = 10 ; m = 100 ; mse = 136.02500256276352 ; timp = 0.057
# incercare = 21 ; p = 10 ; m = 150 ; mse = 134.18976904946422 ; timp = 0.055
# incercare = 22 ; p = 10 ; m = 200 ; mse = 131.07659317656186 ; timp = 0.053
# incercare = 23 ; p = 10 ; m = 300 ; mse = 128.84499379461766 ; timp = 0.05
# incercare = 24 ; p = 15 ; m = 50 ; mse = 167.23865928788578 ; timp = 0.12
# incercare = 25 ; p = 15 ; m = 100 ; mse = 139.3630103123642 ; timp = 0.12
# incercare = 26 ; p = 15 ; m = 150 ; mse = 137.8625965054922 ; timp = 0.12
# incercare = 27 ; p = 15 ; m = 200 ; mse = 133.3353575722042 ; timp = 0.1
# incercare = 28 ; p = 15 ; m = 300 ; mse = 130.23212848287844 ; timp = 0.094
# incercare = 29 ; p = 20 ; m = 50 ; mse = 171.57004380794862 ; timp = 0.2
# incercare = 30 ; p = 20 ; m = 100 ; mse = 142.45291113046662 ; timp = 0.18
# incercare = 31 ; p = 20 ; m = 150 ; mse = 138.20998135006838 ; timp = 0.17
# incercare = 32 ; p = 20 ; m = 200 ; mse = 132.94100207667714 ; timp = 0.16
# incercare = 33 ; p = 20 ; m = 300 ; mse = 129.2144044668986 ; timp = 0.15
# incercare = 34 ; p = 25 ; m = 50 ; mse = 175.3584664418053 ; timp = 0.29
# incercare = 35 ; p = 25 ; m = 100 ; mse = 145.7395813576579 ; timp = 0.28
# incercare = 36 ; p = 25 ; m = 150 ; mse = 142.18435300423477 ; timp = 0.28
# incercare = 37 ; p = 25 ; m = 200 ; mse = 136.17241699486237 ; timp = 0.25
# incercare = 38 ; p = 25 ; m = 300 ; mse = 131.31613203177383 ; timp = 0.22
# incercare = 39 ; p = 30 ; m = 50 ; mse = 175.18906567761883 ; timp = 0.41
# incercare = 40 ; p = 30 ; m = 100 ; mse = 144.7825730424715 ; timp = 0.4
# incercare = 41 ; p = 30 ; m = 150 ; mse = 140.79206618581432 ; timp = 0.38
# incercare = 42 ; p = 30 ; m = 200 ; mse = 134.43975296162031 ; timp = 0.35
# incercare = 43 ; p = 30 ; m = 300 ; mse = 129.388363365949 ; timp = 0.32
# incercare = 44 ; p = 40 ; m = 50 ; mse = 172.93444437221098 ; timp = 0.73
# incercare = 45 ; p = 40 ; m = 100 ; mse = 143.2247188376653 ; timp = 0.69
# incercare = 46 ; p = 40 ; m = 150 ; mse = 139.4242638548078 ; timp = 0.65
# incercare = 47 ; p = 40 ; m = 200 ; mse = 133.30974076282794 ; timp = 0.62
# incercare = 48 ; p = 40 ; m = 300 ; mse = 129.61469328056435 ; timp = 0.55
# incercare = 49 ; p = 50 ; m = 100 ; mse = 145.2610938983776 ; timp = 1.1
# incercare = 50 ; p = 50 ; m = 150 ; mse = 141.39801828018366 ; timp = 1.0
# incercare = 51 ; p = 50 ; m = 200 ; mse = 135.29715840973924 ; timp = 1.0
# incercare = 52 ; p = 50 ; m = 300 ; mse = 130.00354919507615 ; timp = 0.87
# incercare = 53 ; p = 60 ; m = 100 ; mse = 145.0435193052482 ; timp = 1.6
# incercare = 54 ; p = 60 ; m = 150 ; mse = 144.31981396771928 ; timp = 1.5
# incercare = 55 ; p = 60 ; m = 200 ; mse = 138.02948441451434 ; timp = 1.4
# incercare = 56 ; p = 60 ; m = 300 ; mse = 132.16849393708787 ; timp = 1.2
# incercare = 57 ; p = 70 ; m = 100 ; mse = 145.16854459288916 ; timp = 2.1
# incercare = 58 ; p = 70 ; m = 150 ; mse = 145.378791838654 ; timp = 2.0
# incercare = 59 ; p = 70 ; m = 200 ; mse = 140.03663550478052 ; timp = 1.9
# incercare = 60 ; p = 70 ; m = 300 ; mse = 133.1338655741292 ; timp = 1.6
# incercare = 61 ; p = 100 ; m = 150 ; mse = 153.15846378758934 ; timp = 4.0
# incercare = 62 ; p = 100 ; m = 200 ; mse = 149.9318810848371 ; timp = 3.8
# incercare = 63 ; p = 100 ; m = 300 ; mse = 139.65062740859779 ; timp = 3.3

