import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime


#componenta continua = prima componenta din fft

# train.csv 18288 esantioane din ora in ora
# frecventa 1/ora sau 1/60 pe minut sau 1/360 pe secunda, 1/360[Hz]

# - 25-09-2014 23:00 - 25-08-2012 00:00 






def interval_ocupat(csv):
    data_inceput = "25-09-2014 23:00"
    data_final = "25-08-2012 00:00"
    fmt = "%d-%m-%Y %H:%M"
    t1 = datetime.datetime.strptime(data_inceput,fmt)
    t2 = datetime.datetime.strptime(data_final,fmt)

    delta = t1-t2
    print(str(delta.days/365)+" ani sau")
    print(str(delta.days)+" zile sau")
    print(str(delta.seconds)+" secunde")



def citire(csv='data/Train.csv'):

    dtype=[(''),(),()]

    x = np.genfromtxt(csv,dtype='u8,U,u8', delimiter=',',skip_header=1,like=np.array)
    return x

def main():
    x = citire()
    x = np.fft.fft(x)
    for i in range(5):
        print (x[i])
    

    print(np.shape(x))
    print(x.dtype)
    print(interval_ocupat(citire()))
    
    # 1- 1/T
main()