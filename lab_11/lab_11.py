import numpy as np
import matplotlib.pyplot as plt



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


# ex_2
def hankelizeaza(serie, L):
    N = len(serie)

    K = N - L + 1

    X = np.zeros((L,K))

    for i in range(L):
        X[i] = serie[i : i + K]
    
    return X


def ex_3(N=1000,L = 70):
    timp, y, trend, sezon, zgomot = ex_a(N=N, plot=False)
    
    X = hankelizeaza(y, L)

    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    

    X_Xt = X @ X.T
    Xt_X = X.T @ X


    #valori proprii
    X_Xt_val_prop, X_Xt_vec_prop = np.linalg.eigh(X_Xt)
    Xt_X_val_prop, Xt_X_vec_prop = np.linalg.eigh(Xt_X)
    


    X_Xt_val_prop_sorted = X_Xt_val_prop[::-1]
    Xt_X_val_prop_sorted = Xt_X_val_prop[::-1]
    X_Xt_vec_prop_sorted = X_Xt_vec_prop[:, ::-1]
    Xt_X_vec_prop_sorted = Xt_X_vec_prop[:, ::-1]

    # verificam X @ X^T == U @ Sigma^2 @ U^T
    verificare_1 = np.allclose(X_Xt,U @ np.diag(sigma**2) @ U.T)

    # Verificam Valorile proprii X @ X^T sunt egale cu sigma^2
    verificare_2 = np.allclose(X_Xt_val_prop_sorted, sigma**2)

    # Verificam vectorii proprii X @ X^T sunt egali cu U
    verificare_3 = np.allclose(np.abs(X_Xt_vec_prop_sorted), np.abs(U))
    print(f"Verificam daca X @ X^T == U @ Sigma^2 @ U^T : {verificare_1}")
    print(f"Verificam daca Valorile proprii X @ X^T sunt egale cu sigma^2: {verificare_2}")
    print(f"Verificam daca vectorii proprii X @ X^T sunt egali cu U:{verificare_3}")


def dez_hankelizeaza(Matrice, N=1000):


    L, K = Matrice.shape

    matrice_oglindita = np.fliplr(Matrice)

    start = K -1
    stop = -L

    serie = [np.mean(matrice_oglindita.diagonal(k)) for k in range(start, stop, -1)]
    return serie

def ex_4(N=1000, L = 70):
    timp, y, trend_original, sezon_original, zgomot_original = ex_a(N=N, plot=False)

    X = hankelizeaza(y,L)

    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

    componente_ssa = []

    for i in range(len(sigma)):
        Xi = sigma[i] * np.outer(U[:, i], Vt[i, :])

        componente_ssa.append(dez_hankelizeaza(Xi, N))

    componente_ssa = np.array(componente_ssa)

    ssa_trend = componente_ssa[0]

    ssa_sezon = np.sum(componente_ssa[1:5], axis=0)

    ssa_y = np.sum(componente_ssa, axis=0)




    fig, axs = plt.subplots(3, figsize=(10, 12), sharex=True)
    
    axs[0].plot(timp, y, label="Serie originala", color="black")
    axs[0].plot(timp, ssa_y, label="Reconstructie", color="red",linestyle = '--',alpha = 0.5)
    axs[0].set_title(f"Serie originala vs reconstructie")
    axs[0].legend()
    


    axs[1].plot(timp, trend_original, label="Trend original", color="black")
    axs[1].plot(timp, ssa_trend, label="Trend reconstructie", color="red")
    axs[1].set_title("Trend original vs reconstructie")
    axs[1].legend()



    axs[2].plot(timp, sezon_original, label="Sezon original", color="black")
    axs[2].plot(timp, ssa_sezon, label="Sezon reconstruit", color="red")
    axs[2].set_title("Sezon original vs reconstructie")
    axs[2].legend()

    plt.savefig("imagini/ex_4.pdf")
    plt.show()

ex_4()