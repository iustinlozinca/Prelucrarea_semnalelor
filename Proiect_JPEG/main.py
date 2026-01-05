import os
import time
import matplotlib.pyplot as plt

import jpeg
import MJPEG


def set_title():
    print("=" * 60)
    print("Proiectul meu, Lozinca Iustin Florin 343")
    print("=" * 60)


def curata_ecran():
    os.system('clear')
    set_title

def seteaza_imagine():
    curata_ecran()
    print(f"Imagine curenta selectata: {jpeg.nume_poza}")
    print("-" * 60)
    noua_cale = input("Introdu calea catre noua imagine (ex: test.png): ").strip()
    
    if os.path.isfile(noua_cale):
        jpeg.nume_poza = noua_cale
        print(f"\nImaginea a fost schimbata in: {noua_cale}")
    else:
        print(f"\nFisierul '{noua_cale}' nu exista!")
    
    input("\nApasa Enter ca sa te intorci")

def demo_ycbcr():
    try:
        jpeg.test_ycbcr_conversion()
    except Exception as e:
        print(f"Probleme!! {e}")
    input("Apasa enter")

def demo_jpeg_huffman():
    print(f"\n Se comprima imaginea: {jpeg.nume_poza}")
    try:
        jpeg.test_jpg(huffman=True)
    except Exception as e:
        print(f"Probleme!{e}")
    input("Apasa Enter")

def demo_mse_target():
    mse = input("Introdu Target MSE")
    print(f"\nSe cauta factorul de compresie pentru {jpeg.nume_poza}...")
    try:
        jpeg.test_mse_target(mse)
    except Exception as e:
        print(f"Probleme la demo mse {e}")
    input("Apasa enter...")

def demo_video():
    curata_ecran()
    print("--- COMPRESIE VIDEO")
    
    input_vid = input(f"Video intrare [{MJPEG.default_cale_input}]: (scrie un input sau apasa enter) ").strip()
    if not input_vid:
        input_vid = MJPEG.default_cale_input
        
    output_vid = input(f"Video iesire [{MJPEG.default_cale_output}]: (scrie un output sau apasa enter) ").strip()
    if not output_vid:
        output_vid = MJPEG.default_cale_output
        
    nr_frames = input("Numar de cadre (Default= 50, 'tot' pentru tot): ").strip()
    
    max_f = 50
    if nr_frames.lower() == 'tot':
        max_f = 0
    elif nr_frames.isdigit():
        max_f = int(nr_frames)

    if not os.path.exists(input_vid):
        print(f"\nNu gasesc clipul {input_vid}")
        input("Apasa Enter...")
        return

    print("\nINCEPEM!!!!!")
    try:
        MJPEG.proceseaza_video(cale_input=input_vid, cale_output=output_vid, max_frames=max_f)
        print("\nGataa!!!")
    except Exception as e:
        print(f"\nAu aparut niste probleme: {e}")
    
    input("\nApasa Enter pentru a te intoarce")

def meniu():
    while True:
        curata_ecran()
        set_title()
        print(f"Fisier Imagine Activ: [ {jpeg.nume_poza} ]")
        print("-" * 60)
        print("1. Vizualizare conversie RGB -> YCBCR")
        print("2. Test Compresie JPEG (cu Huffman!!) - Vizual")
        print("3. Test Optimizare MSE")
        print("4. Procesare Video")
        print("5. Schimba imaginea")
        print("0. Iesire")
        print("-" * 60)
        
        optiune = input("Alege o optiune: ").strip()

        if optiune == "1":
            demo_ycbcr()
        elif optiune == "2":
            demo_jpeg_huffman()
        elif optiune == "3":
            demo_mse_target()
        elif optiune == "4":
            demo_video()
        elif optiune == "5":
            seteaza_imagine()
        elif optiune == "0":
            print("\n Va multumesc pentru atentie!")
            break
        else:
            print("\nOptiune invalida!")
            time.sleep(1)

meniu()


