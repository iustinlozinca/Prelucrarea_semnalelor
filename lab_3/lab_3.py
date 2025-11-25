from pprint import pprint
import math as math
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as anim

#exemplu sintaxa math.e**(1j*5)
# semnal discret x[n]

def fourier(N):
    matrice = []
    for v in range(N):
        matrice.append([])
        for w in range(N):
            matrice[v].append(np.exp(w*v/N*(-2)*np.pi*1j))
    return matrice

def ex_1(N=8):

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
    plt.savefig("imagini/ex_1.pdf", format = 'pdf')

         
def ex_2(Figura,animate):

    def Figura_1(animate = False):
        timp = np.linspace(0,4,1000)
        x = lambda t: np.sin(2*np.pi* 2 * t)
        y = lambda t: x(t)*np.exp(-1j*np.pi*t)

        fig, axs = plt.subplots(1,2)


        axs[0].plot(timp,x(timp))
        axs[1].set_xlim(-1,1)
        axs[1].set_ylim(-1,1)
        plt.subplots_adjust(wspace= 0.5)

        match animate:

            case False:
                y_d = y(timp)
                distanta = np.abs(y_d)                    
                               
                axs[1].scatter(y_d.real,y_d.imag, c = distanta,
                                    cmap = 'viridis'
                                   , s = 10)             
                axs[1].set_xlabel("real")
                axs[1].set_ylabel("imaginar")
                axs[0].plot(timp,x(timp))

                plt.savefig("imagini/ex_2_figura_1.pdf", format = "pdf")
                plt.show()

            case True:

                y_d_real = []
                y_d_imag = []
                distanta = []
                for t in timp:
                    axs[1].cla()
                    axs[0].cla()
                    axs[1].set_xlim(-1,1)
                    axs[1].set_ylim(-1,1)
                    axs[0].stem(t,x(t))
                    y_d_real.append(y(t).real)
                    y_d_imag.append(y(t).imag)
                    distanta.append(np.abs(y(t)))
                    axs[1].scatter(y_d_real,y_d_imag, c = distanta,
                                    # cmap = 'plasma'
                                    # cmap = 'jet'
                                    cmap = 'viridis'
                                   , s = 10)
                    axs[1].set_xlabel("real")
                    axs[1].set_ylabel("imaginar")
                    axs[1].plot(y(t).real,y(t).imag,'bo')
                    axs[0].plot(timp,x(timp))
                    plt.pause(0.00001)             
        
    
    def Figura_2(animate = False):

        # z[w] = x[n]e^{-2 pi j w n}
        timp = np.linspace(0,4,1000)
        x = lambda t: np.sin(2*np.pi* 2 * t)
        z = lambda t,w: x(t)*np.exp(-2*1j*np.pi*w*t)

        W = [1,2,5,7,8]        
        plots_number = float(len(W))

        fig, axs = plt.subplots(int(np.ceil(plots_number/2)),int(np.floor(plots_number/2)))

        plt.subplots_adjust(hspace= 0.5)
        plt.subplots_adjust(wspace= 0.5)

        # for ax in axs.flatten():
        #     ax.set_xlim(-1,1)
        #     ax.set_ylim(-1,1)



        match animate:
            case False:
                for i, ax in enumerate(axs.flatten()):
                    if i >= len(W):
                        ax.axis('off')
                        continue

                    ax.set_ylim(-1,1)
                    ax.set_xlim(-1,1)

                    ax.axhline(0, color='black', linewidth=1, zorder=-1)
                    ax.axvline(0, color='black', linewidth=1, zorder=-1)

                    w = W[i]

                    z_d= z(timp,w)
                    
                    distanta = np.abs(z_d)

                    ax.scatter(z_d.real,
                               z_d.imag,
                               c = distanta,
                               cmap = 'viridis',
                               s = 10
                               )
                    
                    ax.set_title(f"w = {w}")
                    ax.set_xlabel("real")
                    ax.set_ylabel("imaginar")

                plt.savefig("imagini/ex_2_figura_2.pdf" , format = "pdf")
                plt.show()



            case True:
                z_d_real = [[] for _ in W]
                z_d_imag = [[] for _ in W]
                distanta = [[] for _ in W]


                for t in timp:
                    for i, ax in enumerate(axs.flatten()):

                        if i >= len(W):
                            ax.axis('off')
                            continue


                        ax.cla()
                        ax.set_ylim(-1,1)
                        ax.set_xlim(-1,1)

                        ax.axhline(0, color='black', linewidth=1, zorder=-1)
                        ax.axvline(0, color='black', linewidth=1, zorder=-1)


                        w = W[i]

                        val = z(t,w)

                        z_d_real[i].append(val.real)
                        z_d_imag[i].append(val.imag)

                        distanta[i].append(np.abs(val))


                        ax.scatter(z_d_real[i],
                                   z_d_imag[i],
                                   c = distanta[i],
                                   cmap = 'viridis',
                                   s = 10
                                   
                                   )
                        
                        ax.set_title(f"w = {w}")
                        ax.set_xlabel("real")
                        ax.set_ylabel("imaginar")


                    plt.pause(0.0000001)




    match Figura:
        case 0 | 1 :
            Figura_1(animate)
        case _ :
            Figura_2(animate)



def ex_3(N=500):


    timp = np.linspace(0,3,N)

    frecvente = np.linspace(0,50,500)

    x = lambda t: np.sin(2*np.pi*10*t+2)+3*np.sin(2*np.pi*3*t)+4*np.sin(2*np.pi*t*5+4)

    X = lambda w: sum([x(t)*np.exp(-2*np.pi*1j*t*w) for t in timp])

    fig , axs = plt.subplots(2)

    plt.subplots_adjust(hspace = 0.7)

    axs[0].plot(timp,x(timp))
    axs[0].set_xlabel('Timp(s)')
    axs[0].set_ylabel('x(t)')

    axs[1].stem(frecvente,np.abs(X(frecvente)))
    axs[1].set_xlabel('Frecventa (Hz)')
    axs[1].set_ylabel('|X(w)|')

    plt.savefig("imagini/ex_3.pdf" , format = "pdf")
    plt.show()

def main():
    ex_1()
    ex_2(1,False)
    ex_2(2,False)
    ex_3()

main()