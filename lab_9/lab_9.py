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



def ex_2(L = 50, data = ex_a(plot= False)[1],plot = False):
        
    def simple_smoothing(params, data, return_series= False):

        try:
            _ , alpha = params
        except TypeError:

            alpha = params

        n = len(data)
        s = np.zeros(n)
        predictie = np.zeros(n)

        s[0] = data[0]
        predictie[0] = data[0]

        for t in range(1, n):
            predictie[t] = s[t-1]
            s[t] = alpha * data[t] + (1 - alpha) * s[t-1]

        if return_series:
            return predictie
        return np.mean((data[1:] - predictie[1:])**2)
    

    def double_smoothing(params, data, return_series = False):

        if len(params) == 3:
            _ , alpha,beta = params
        elif len(params) == 2:
            alpha, beta = params
        else:
            print('double smoothing params incorect')
            return


        n = len(data)
        s = np.zeros(n)
        b = np.zeros(n)
        predictie = np.zeros(n)

        s[0] = data[0]
        b[0] = data[1] - data[0]
        predictie[0] = data[0]

        for t in range(1, n):
            predictie[t] = s[t-1] + b[t-1]

            old_s = s[t-1]
            s[t] = alpha * data[t] + (1 - alpha) * (s[t-1] + b[t-1])
            b[t] = beta * (s[t] - old_s) + (1 - beta) * b[t-1]

        if return_series:
            return predictie
        else:
            return np.mean((data[1:] - predictie[1:])**2)
        
    def triple_smoothing(params, data, L=L, return_series=False):

        if len(params) == 4:
            _ , alpha,beta, gamma = params
        elif len(params) == 3:
            alpha, beta, gamma = params
        else:
            print('triple smoothing params incorect')
            return


        n = len(data)
        s = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        predictie = np.zeros(n)

        s[L-1] = np.mean(data[:L])
        b[L-1] = (data[L] - data[0]) / L

        for i in range(L):
            c[i] = data[i] - s[L-1]
            predictie[i] = data[i]

        for t in range(L,n):
            predictie[t] = s[t-1] + b[t-1] + c[t-L]

            old_s = s[t-1]

            s[t] = alpha * (data[t] - c[t-L]) + (1-alpha)*(s[t-1] + b[t-1])
            b[t] = beta * (s[t] - old_s) + (1- beta) * b[t-1]
            c[t] = gamma * (data[t] - s[t] - b[t-1]) + (1-gamma) * c[t-L]


        if return_series:
            return predictie
        else:
            return np.mean((data[L:] - predictie[L:])**2)

    valori = np.linspace(0.001, 0.99 , 5)

    best_mse = float('inf')
    best_alpha = None

    for alpha in valori:
        mse = simple_smoothing(alpha, data)
        if mse <= best_mse:
            best_mse = mse
            best_alpha = alpha

    best_simple = (best_mse,best_alpha)

    best_mse = float('inf')
    best_alpha = None
    best_beta = None

    for alpha in valori:
        for beta in valori:
            mse = double_smoothing((alpha,beta),data)
            if mse <= best_mse:
                best_mse = mse
                best_alpha = alpha
                best_beta = beta

    best_double = (best_mse,best_alpha,best_beta)

    best_mse = float('inf')
    best_alpha = None
    best_beta = None
    best_gamma = None

    for alpha in valori:
        for beta in valori:
            for gamma in valori:
                mse = triple_smoothing((alpha,beta,gamma),data)
                if mse <= best_mse:
                        best_mse = mse
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma

    best_triple = (best_mse,best_alpha,best_beta,best_gamma)

    if plot:
        
        bests = [best_simple,best_double,best_triple]

        y_simple = simple_smoothing(best_simple,data,return_series= True)

        y_double = double_smoothing(best_double,data,return_series=True)

        y_triple = triple_smoothing(best_triple,data,return_series=True)



        fig, axs = plt.subplots(5)
        plt.subplots_adjust(hspace = 2)


        axs[0].plot(data)

        axs[1].plot(data)
        axs[1].plot(y_simple)
        axs[1].set_title(f'simpla, mse ={bests[0][0]}')



        axs[2].plot(data)
        axs[2].plot(y_double)
        axs[2].set_title(f'dubla, mse ={bests[1][0]}')

        axs[3].plot(data)
        axs[3].plot(y_triple)
        axs[3].set_title(f'tripla, mse ={bests[2][0]}')


        axs[4].plot(data)
        axs[4].plot(y_simple)
        axs[4].plot(y_double)
        axs[4].plot(y_triple)
        axs[4].set_title('Toate')

        plt.savefig('imagini/ex_2.pdf')
        plt.show()



    else:
        return best_simple,best_double,best_triple


def main():
    pass
main()
