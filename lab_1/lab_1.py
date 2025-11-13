import numpy as np
import matplotlib.pyplot as plt
from functii import *
pi = np.pi

### functii
# x(t) = cos(520πt + π/3)
# y(t) = cos(280πt − π/3)
# z(t) = cos(120πt + π/3)

def x(t): return np.cos(520*pi*t + pi/3)

def y(t): return np.cos(280*pi*t - pi/3)

def z(t): return np.cos(120*pi*t + pi/3)

def timp(x = 0.005): return np.linspace(0,x,600)

#punctul c -> betisoare (stem)
#punctul b -> plot (linie continua)


def ex_1_b():
    fig, axs = plt.subplots(3)
    fig.suptitle("execitiu_1_b")
    plt.subplots_adjust(hspace=0.7)
    axs[0].plot(timp(0.02),[x(t) for t in timp(0.02)])
    axs[0].set_title("x")
    axs[1].plot(timp(0.02),[y(t) for t in timp(0.02)])
    axs[1].set_title("y")
    axs[2].plot(timp(0.02),[z(t) for t in timp(0.02)])
    axs[2].set_title("z", pad = 10)
    plt.savefig(save_folder("ex_1_b.pdf"), format='pdf')
    plt.show()


def ex_1_c():
    fs = 300  #frecventa
    Ts = 1/fs  #perioada
    durata = 0.03  #durata in secunde
    
    n = np.arange(int(durata*fs))  #indice esantioane
    t_n = n * Ts  #valoarea timpului la esantioane

    x_n = x(t_n)
    y_n = y(t_n)
    z_n = z(t_n)

    fig, axs = plt.subplots(3)
    plt.subplots_adjust(hspace=0.7)

    axs[0].stem(n, x_n)
    axs[0].set_title('x')
    
    axs[1].stem(n, y_n)
    axs[1].set_title('y')
    
    axs[2].stem(n, z_n)
    axs[2].set_title('Z')
    plt.savefig(save_folder("ex_1_c.pdf"), format='pdf')
    plt.show()
    print('3')

    
def ex_2():

    def sinus(x,frecventa):
        return np.sin(2*pi*x*frecventa)
    timp = []
    fig, axs = plt.subplots(5)
    plt.subplots_adjust(hspace=0.9)
    # (a)
    i = 0
    axs[i].set_title("400hz , 1600 esantioane")
    timp.append(np.linspace(0, 0.01, 1600))
    axs[i].plot(timp[i],sinus(timp[i],400))
    
    # (b)
    i = 1
    axs[i].set_title("b")
    timp.append(np.linspace(0, 3, 300)  )  
    axs[i].plot(timp[i],sinus(timp[i],800))

    # (c)
    i = 2
    axs[i].set_title("c")
    timp.append(np.linspace(0, 5, 170)  )
    axs[i].set_ylim(-1,1)
    axs[i].plot(timp[i],np.floor(sinus(timp[i],120)))

    # (d)
    i = 3
    axs[i].set_title("d")
    timp.append(np.linspace(0, 4, 150))  
    axs[i].plot(timp[i],np.sign(sinus(timp[i],300)))





    plt.savefig(save_folder("ex_2.pdf"), format='pdf')
    plt.show()


def main():
    ex_2()
main()