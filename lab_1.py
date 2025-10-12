import numpy as np
import matplotlib.pyplot as plt

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

def timp():
    return np.linspace(0,0.003,600)

#punctul c -> betisoare (stem)
#punctul b -> plot (linie continua)

def grafice_ex_1_b():
    plt.plot(timp(),[x(t) for t in timp()])
    plt.savefig("test.eps", format = 'eps')
    plt.show()

def grafice_ex_1_c():
    plt.stem(timp(),[x(t) for t in timp()])
    plt.savefig("testz.eps", format = 'eps')
    plt.show()


def main():
    grafice_ex_1_c()
main()