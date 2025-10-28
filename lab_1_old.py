import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


output_imagini = os.path.join("imagini")
os.makedirs(output_imagini, exist_ok=True)

pi = np.pi

### functii

# x(t) = cos(520πt + π/3)
# y(t) = cos(280πt − π/3)
# z(t) = cos(120πt + π/3)

def x(t):
    return np.cos(520*pi*t + pi/3)

def y(t):
    return np.cos(280*pi*t - pi/3)

def z(t):
    return np.cos(120*pi*t + pi/3)

def sinus(x):
    return np.sin(x)

def timp(x = 0.005):
    return np.linspace(0,x,600)

def save_folder(nume_file):
    return os.path.join(output_imagini,nume_file)

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
    plt.savefig(save_folder("ex_1_b.eps"), format='eps')


def ex_1_c():
    fs = 200  #frecventa
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
    print('1')
    plt.savefig(save_folder("ex_1_c.eps"), format='eps')
    print('2')
    plt.show()
    print('3')


def ex_2():
    def sinusoid(x,frecv,timp):
        return np.sin(x*pi*frecv*timp)


    frecventa = 1
    timp = 1600/400
    nr_esantioane = timp*frecventa

    fig, axs = plt.subplots(3)
    plt.subplots_adjust(hspace=0.7)



    plt.savefig(save_folder("ex_1_c.eps"), format='eps')

def main():
    ex_1_c()
main()