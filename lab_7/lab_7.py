from scipy import ndimage
import scipy.datasets
import numpy as np
import matplotlib.pyplot as plt


def SNR(original,zgomotos):
    mse = np.mean((original-zgomotos)**2)
    if mse == 0:
        return float("inf")
    var_original = np.var(original)
    snr = 10*np.log10(var_original/mse)
    return snr

def plot_signal_spectrum(x,titlu):
    Y = np.fft.fft2(x)
    Y_shifted = np.fft.fftshift(Y) #fftshit folosit pentru a muta frecventa 0  in centru
    freq_db = 20 * np.log10(np.abs(Y_shifted) + 1e-9)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(freq_db, cmap='inferno')
    plt.title('Spectru')
    plt.colorbar()

    plt.savefig(titlu)
    plt.show()

def reconstruct_from_spectrum(coords_list, N, titlu):
    Y = np.zeros((N, N), dtype=complex)
    for (r, c) in coords_list:
        Y[r, c] = 1
        
    x_rec = np.fft.ifft2(Y)
    x_rec = np.real(x_rec)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(x_rec, cmap='gray')
    plt.axis('off')

    plt.savefig(titlu)
    plt.show()

def ex_1(N = 128):

    n1 = np.linspace(0,1,N)
    n2 = np.linspace(0,1,N)
    n1_grid , n2_grid = np.meshgrid(n1, n2)

    x1 = lambda n1,n2: np.sin(2*np.pi*n1+3*np.pi*n2)
    x2 = lambda n1,n2: np.sin(4 *np.pi*n1) + np.cos(6*np.pi*n2)

    plot_signal_spectrum(x1(n1_grid,n2_grid),"imagini/ex_1/1.pdf")

    plot_signal_spectrum(x2(n1_grid,n2_grid),"imagini/ex_1/2.pdf")

    reconstruct_from_spectrum([(0, 5), (0, N-5)],N,"imagini/ex_1/3.pdf")

    reconstruct_from_spectrum([(5, 0), (N-5, 0)],N,"imagini/ex_1/4.pdf")

    reconstruct_from_spectrum([(5, 5), (N-5, N-5)],N,"imagini/ex_1/5.pdf")






def low_pass_din_fft(imagine_fft, matrice, raza_taiere):
    """
    Luam fft-ul imaginii si raza de taiere si
    taiem incat sa ramana un cerc in interior
    """
    masca = matrice <= raza_taiere**2

    return imagine_fft * masca


def comprima_target_snr(imagine, snr_target, max_iteratii = 10):
    linii_nr, coloane_nr = imagine.shape

    y, x = np.ogrid[:linii_nr, :coloane_nr]

    matrice_distanta = (x-coloane_nr//2)**2 + (y-linii_nr//2)**2


    fft_original = np.fft.fft2(imagine)
    fft_shifted = np.fft.fftshift(fft_original)

    low = 0
    high = np.sqrt((linii_nr//2)**2 + (coloane_nr//2)**2)

    best_imagine = None
    best_snr = 0
    best_raza = high

    for i in range(max_iteratii):
        mid_raza = (low+high)/2

        fft_filtrat = low_pass_din_fft(fft_shifted, matrice_distanta, mid_raza)

        imagine_noua = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtrat)))

        snr_curent = SNR(imagine,imagine_noua)

        if snr_curent <snr_target:
            low = mid_raza
        else:
            best_imagine = imagine_noua
            best_snr = snr_curent
            best_raza = mid_raza
            high = mid_raza

    return best_imagine, best_snr

def ex_2(snr_target = 5):

    raton = scipy.datasets.face(gray=True)

    rezultat , best_snr = comprima_target_snr(raton,snr_target)

    if best_snr == 0:
        print("Ceva a mers prost, probabil SNR nerealist")
    
    else:

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(raton, cmap = 'gray')
        plt.title("Ratonul original")

        plt.subplot(1,2,2)
        plt.imshow(rezultat, cmap = "gray")
        plt.title(f"Noul Raton, SNR:{best_snr:.2f}")

        plt.savefig("imagini/ex_2.pdf")
        plt.show()

def ex_3(pixel_noise = 200):

    raton = scipy.datasets.face(gray = True).astype(float)

    zgomot = np.random.randint(-pixel_noise, high = pixel_noise+1, size = raton.shape)

    raton_agitat = raton + zgomot

    raton_agitat = np.clip(raton_agitat, 0 , 255)

    snr_initial = SNR(raton, raton_agitat)

    linii_nr, coloane_nr = raton.shape


    y, x = np.ogrid[:linii_nr, :coloane_nr]

    matrice_distanta = (x-coloane_nr//2)**2 + (y-linii_nr//2)**2

    fft_zgomotos = np.fft.fftshift(np.fft.fft2(raton_agitat))

    best_snr = 0
    best_raton = None
    best_raza = 0


    for raza in range(10,100,5):
        fft_filtrat = low_pass_din_fft(fft_zgomotos, matrice_distanta, raza)

        raton_nou = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtrat)))

        snr_curent = SNR(raton, raton_nou)

        if snr_curent > best_snr:
            best_snr = snr_curent
            best_raton = raton_nou
            best_raza = raza


    print(f"SNR original:{snr_initial} \n SNR obtinut:{best_snr}")
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(raton_agitat, cmap = 'gray')
    plt.title(f"Ratonul agitat, SNR ={snr_initial}")

    plt.subplot(1,2,2)
    plt.imshow(best_raton, cmap = "gray")
    plt.title(f"Noul Raton, SNR ={best_snr}")
    
    plt.savefig("imagini/ex_3.pdf")
    plt.show()


def main():
    ex_1()
    ex_2()
    ex_3()

main()
