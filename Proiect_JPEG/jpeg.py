import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn

from huffman import HuffmanCoder, ZIGZAG_ORDER, zigzag_flatten, inverse_zigzag


# Surse:
# https://en.wikipedia.org/wiki/YCbCr
# https://www.geeksforgeeks.org/dsa/huffman-coding-in-python/


# Magic numbers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Quantization tables

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# Standarde Kr,      Kg,     Kb
BT601 = (  0.299 ,  0.587,  0.114  )

BT709 = (  0.2126 , 0.7152, 0.0722 ) 

BT2020 = ( 0.267,   0.6780, 0.0593 )


# Variabile globale

default_standard = BT2020

nume_poza = "test.png"

default_dctn_type = 2

default_norm_type = "ortho"

def color_matrix(standard= default_standard):
    Kr , Kg, Kb = standard

    matrix = np.array([
        [Kr,Kg,Kb],
        [-0.5 * (Kr/(1-Kb)), -0.5*(Kg / (1- Kb)), 0.5],
        [0.5, -0.5 * (Kg / (1 - Kr)), -0.5 * (Kb / ( 1 - Kr))]

        ])

    inverse = np.array([
        [1 , 0 , 2- 2 * Kr],
        [1 , -(Kb / Kg)*(2 - 2 * Kb), -(Kr/Kg)*(2-2*Kr)],
        [1 , 2 - 2* Kb , 0]
        
        
        ])

    return matrix, inverse


def rgb_to_ycbcr(img , standard = default_standard):
    
    color_mat , _ = color_matrix(standard)
    img_float = img.astype(np.float64)

    if img_float.max() <= 1.0 and img_float.max() > 0:
        img_float *= 255.0


    ycbcr = img_float.dot(color_mat.T) # inmultirea arata asa ciudat din cauza shapeurilor

    ycbcr[:, :, 1] += 128
    ycbcr[:, :, 2] += 128

    return ycbcr

def ycbcr_to_rgb(img, standard = default_standard):
    
    _ , color_mat = color_matrix(standard)
    rgb = img.astype(np.float64)

    rgb[:, :, 1] -= 128
    rgb[:, :, 2] -= 128

    rgb = rgb.dot(color_mat.T)


    np.clip(rgb, 0 , 255, out = rgb)

    return rgb.astype(np.uint8)

def test_ycbcr_conversion():
    poza = plt.imread(nume_poza)

    poza_ycbcr = rgb_to_ycbcr(poza)
    poza_rgb = ycbcr_to_rgb(poza_ycbcr)


    if poza.dtype != np.uint8:
        image_input_uint8 = (poza * 255).astype(np.uint8) if poza.max() <= 1.0 else poza.astype(np.uint8)
    else:
        image_input_uint8 = poza

    diff = np.mean((image_input_uint8 - poza_rgb) ** 2)

    plt.subplot(221)
    plt.imshow(poza)
    plt.title("Poza originala")
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(poza_ycbcr[:, :, 0], cmap='jet')
    plt.title("Y din ycbcr")
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(poza_ycbcr[:, :, 0], cmap='grey')
    plt.title("Y din ycbcr alb negru")
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(poza_rgb)
    plt.title(f"(RGB -> ycbr -> RGB), MSE = {diff:.4f}")
    plt.axis('off')        

    plt.tight_layout()
    plt.show()


def pad_img(imagine, pad_mode = "edge"):
    H,W = imagine.shape[:2]

    pad_h =(8 - (H % 8)) % 8
    pad_w =(8 - (W % 8)) % 8

    if imagine.ndim == 3:
        padding = ((0, pad_h),(0,pad_w),(0,0))
    else:
        padding = ((0, pad_h) , (0, pad_w))

    imagine_padded = np.pad(imagine, padding, mode = pad_mode)

    return imagine_padded, (H, W)

def crop(imagine, dimensiune):
    H,W = dimensiune

    return imagine[:H,:W]


def blockify(imagine):

    H, W = imagine.shape
    return imagine.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)

def deblockify(blocks, original_shape):
    H, W = original_shape
    return blocks.transpose(0, 2, 1, 3).reshape(H, W)


def proceseaza_canal(canal ,dctn_type = 2, Q_tabel=Q_jpeg, norm_type = default_norm_type):
    H, W = canal.shape
    
    blocks = blockify(canal)
    
    blocks_dct = dctn(blocks, axes=(2, 3), type=dctn_type, norm=norm_type)
    
    blocks_quant = Q_tabel * np.round(blocks_dct / Q_tabel)
    
    idct_blocks = idctn(blocks_quant, axes=(2, 3), type=dctn_type, norm=norm_type)
    
    return deblockify(idct_blocks, (H, W))



def to_jpg(imagine, Q_luma = Q_jpeg, Q_color = Q_jpeg):


    imagine , dimensiune_originala = pad_img(imagine)
    

    img_ycbcr = rgb_to_ycbcr(imagine)

    img_temp = np.zeros_like(img_ycbcr)

    img_temp[:,:,0] = proceseaza_canal(img_ycbcr[:,:,0], Q_tabel=Q_luma)
    img_temp[:,:,1] = proceseaza_canal(img_ycbcr[:,:,1], Q_tabel=Q_color)
    img_temp[:,:,2] = proceseaza_canal(img_ycbcr[:,:,2], Q_tabel=Q_color)

    img_rezultat = ycbcr_to_rgb(img_temp)

    img_rezultat = crop(img_rezultat , dimensiune_originala)
    return img_rezultat

def proceseaza_canal_huffman(canal ,dctn_type = 2, Q_tabel=Q_jpeg, norm_type = default_norm_type):
    H, W = canal.shape
    
    blocks = blockify(canal)
    
    blocks_dct = dctn(blocks, axes=(2, 3), type=dctn_type, norm=norm_type)
    
    blocks_quant = np.round(blocks_dct / Q_tabel).astype(np.int32)

    coeficienti_zigzag = blocks_quant.reshape(-1, 64)[:, ZIGZAG_ORDER].flatten()
    
    blocks_quant = blocks_quant * Q_tabel

    idct_blocks = idctn(blocks_quant, axes=(2, 3), type=dctn_type, norm=norm_type)
    
    return deblockify(idct_blocks, (H, W)) , coeficienti_zigzag

def comprima_imaginea(imagine, #ycbcr
                      Q_luma = Q_jpeg,
                      Q_color = Q_jpeg
                      ):
    imagine , dimensiune_originala = pad_img(imagine)
    dimensiune_padded = imagine.shape[:2]
    img_ycbcr = rgb_to_ycbcr(imagine)

    coder = HuffmanCoder()

    _, coef_y= proceseaza_canal_huffman(img_ycbcr[:,:,0], Q_tabel=Q_luma)
    _, coef_cb = proceseaza_canal_huffman(img_ycbcr[:,:,1], Q_tabel=Q_color)
    _, coef_cr = proceseaza_canal_huffman(img_ycbcr[:,:,2], Q_tabel=Q_color)

    bits_y, dict_y = coder.compress(coef_y)
    bits_cb, dict_cb = coder.compress(coef_cb)
    bits_cr, dict_cr = coder.compress(coef_cr)    
    
    imagine_comprimata = (bits_y,bits_cb,bits_cr)
    dictionare = (dict_y,dict_cb,dict_cr)

    return imagine_comprimata, dictionare, dimensiune_originala, dimensiune_padded, Q_luma, Q_color

def reconstruie_canal(bits, dictionar, dimensiuni_padded, Q_tabel=Q_jpeg):
    H, W = dimensiuni_padded
    
    coder = HuffmanCoder()
    
    coeficienti_zigzag = coder.decompress(bits, dictionar)
    
    nr_blocuri = coeficienti_zigzag.size // 64
    
    blocuri_zigzag = coeficienti_zigzag.reshape(nr_blocuri, 64)
    
    blocuri_liniare = np.zeros((nr_blocuri, 64), dtype=coeficienti_zigzag.dtype)
    blocuri_liniare[:, ZIGZAG_ORDER] = blocuri_zigzag
    
    blocuri_quant = blocuri_liniare.reshape(nr_blocuri, 8, 8)
    
    blocuri_dct = blocuri_quant * Q_tabel
    
    blocuri_grid = blocuri_dct.reshape(H // 8, W // 8, 8, 8)
    
    idct_blocuri = idctn(blocuri_grid, axes=(2, 3), type=default_dctn_type, norm=default_norm_type)
    
    canal_reconstruit = deblockify(idct_blocuri, (H, W))
    
    return canal_reconstruit



def decomprima_imaginea(imagine_comprimata, dictionare, dimensiune_originala, dimensiune_padded, Q_luma = Q_jpeg, Q_color = Q_jpeg):
    bits_y, bits_cb, bits_cr = imagine_comprimata
    dict_y, dict_cb, dict_cr = dictionare

    coder = HuffmanCoder

    H, W = dimensiune_padded
    
    img_ycbcr_rec = np.zeros((H, W, 3))


    canal_y  = reconstruie_canal(bits_y,  dict_y,  dimensiune_padded, Q_luma)
    canal_cb = reconstruie_canal(bits_cb, dict_cb, dimensiune_padded, Q_color)
    canal_cr = reconstruie_canal(bits_cr, dict_cr, dimensiune_padded, Q_color)

    img_ycbcr_rec[:, :, 0] = canal_y
    img_ycbcr_rec[:, :, 1] = canal_cb
    img_ycbcr_rec[:, :, 2] = canal_cr

    img_rgb = ycbcr_to_rgb(img_ycbcr_rec)

    img_final = crop(img_rgb, dimensiune_originala)

    return img_final

def to_jpg_huffman(imagine, Q_luma=Q_jpeg, Q_color=Q_jpeg):

    rezultate = comprima_imaginea(imagine, Q_luma, Q_color)
    
    (imagine_comprimata, 
     dictionare, 
     dimensiune_originala, 
     dimensiune_padded, 
     q_luma_out, 
     q_color_out) = rezultate

    img_rezultat = decomprima_imaginea(
        imagine_comprimata, 
        dictionare, 
        dimensiune_originala, 
        dimensiune_padded, 
        q_luma_out, 
        q_color_out
    )

    return img_rezultat


def test_jpg(huffman = True):


    img_input = plt.imread(nume_poza)

    if img_input.ndim == 3 and img_input.shape[2] == 4:
        img_input = img_input[:, :, :3] #eliminam alpha

    if img_input.dtype != np.uint8:
        img_input = (img_input * 255).astype(np.uint8)

    if huffman:
        img_compressed = to_jpg_huffman(img_input)
    else:
        img_compressed = to_jpg(img_input)

    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(img_input)
    plt.title("Original (RGB)")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(img_compressed)
    plt.title(f"JPEG Reconstruit")
    plt.axis('off')

    plt.show()

    #Notita: Cu toate ca sunt vizibile artefactele de patratele 8x8, au un efect
    # satisfacator cand dau zoom pe ele, ma simt de parca mi-a reusit corect compresia


def mse(img1, img2):

    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])

    diff = img1[:h, :w].astype(float) - img2[:h, :w].astype(float)

    return np.mean(diff ** 2)

def comprima_target(imagine, mse_tinta, Q_luma = Q_jpeg, Q_color = Q_jpeg, max_iteratii=10):

    low = 0.1
    high = 10.0
    best_alpha = 1.0

    best_rezultat = None
    best_mse = float('inf')

    for i in range(max_iteratii):
        mid = (low + high) / 2
        Q_luma_curent = np.array(Q_luma)*mid
        Q_color_curent = np.array(Q_color)*mid
        imagine_reconstruita = decomprima_imaginea(*comprima_imaginea(imagine, Q_color= Q_color_curent, Q_luma= Q_luma_curent))


        mse_curent = mse(imagine_reconstruita,imagine)


        print(f"Iteratia {i+1}: Factor={mid:.3f}, MSE={mse_curent:.2f}")

        if mse_curent <= mse_tinta:
            best_rezultat = imagine_reconstruita
            best_mse = mse_curent
            best_alpha = mid
            low = mid

        else:
            high = mid

    if best_rezultat is None:
        return imagine_reconstruita
    else:
        print(best_mse)
        return best_rezultat

def test_mse_target(mse_target = 100):
    img_input = plt.imread(nume_poza)
    
    if img_input.ndim == 3 and img_input.shape[2] == 4:
        img_input = img_input[:, :, :3]
    
    if img_input.dtype != np.uint8:
        img_input = (img_input * 255).astype(np.uint8)

    
    img_optim = comprima_target(img_input, mse_tinta=mse_target)

    mse_final = mse(img_input, img_optim)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img_input)
    plt.title("Original")
    
    plt.subplot(122)
    plt.imshow(img_optim)
    plt.title(f"MSE Obtinut: {mse_final:.2f}")
    plt.show()

