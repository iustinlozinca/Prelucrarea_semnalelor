import numpy as np
import matplotlib.pyplot as plt
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
    plt.savefig("imagini/ex_1_b.pdf", format='pdf')
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
    plt.savefig("imagini/ex_1_c.pdf", format='pdf')
    plt.show()

    
def ex_2(ok: bool = True):

    def sinus(t,frecventa):
        return np.sin(2*pi*t*frecventa)
    timp = []
    fig, axs = plt.subplots(5)
    plt.subplots_adjust(hspace=1.5)
    # (a)
    i = 0
    axs[i].set_title("400hz , 1600 esantioane")
    timp.append(np.linspace(0, 0.01, 1600, endpoint= False))
    axs[i].plot(timp[i],a := sinus(timp[i],400))
    
    # (b)
    i = 1
    axs[i].set_title("800hz, 1600 Hz esantionare") #graficul arata urat doar ca daca esantionez mai putin practic nu e acelasi grafic
    timp.append(np.linspace(0, 3, 4800,endpoint= False))  
    axs[i].set_ylim(-1,1)
    axs[i].plot(timp[i],b := sinus(timp[i],800))
    # frecventa = 800 / s
    # 3 secunde => esantioane/3 = esantioane per secunda
    # => 2*800*3= 4800 esantioane /3s
    # Intrebare: de ce valori dupa 4800 arata grafice diferite
    # si de ce 4800 nu arata un grafic


    # (c)
    i = 2
    axs[i].set_title("c")
    timp.append(np.linspace(0, 5, 170, endpoint= False)  )
    axs[i].set_ylim(-1,1)
    axs[i].plot(timp[i],c := np.floor(sinus(timp[i],240)))

    # (d)
    i = 3
    axs[i].set_title("d")
    timp.append(np.linspace(0, 4, 150))  
    axs[i].plot(timp[i],np.sign(sinus(timp[i],300)))


    # (e)

    # i = 4
    # axs[i].set_title("e")
    # arr = np.random.rand(x,y)
    # timp.append(np.linspace())
    if ok:
        plt.savefig("imagini/ex_2.pdf", format='pdf')
        plt.show()

def ex_2_e():
    arr = np.random.rand(128,128)
    plt.imshow(arr)
    plt.savefig("imagini/ex_2_e.pdf",format = "pdf")


def ex_2_f():
    x = np.arange(128)
    y = np.arange(128)
    y = y.reshape(-1,1)
    semnal_2d = x + y
    plt.imshow(semnal_2d)
    plt.savefig("imagini/ex_2_f.pdf",format = "pdf")

def main():
    ex_1_b()
    ex_1_c()
    ex_2()
    ex_2_e()
    ex_2_f()
main()

