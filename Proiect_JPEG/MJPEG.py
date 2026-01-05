import numpy as np
import matplotlib.pyplot as plt
import cv2
from jpeg import comprima_imaginea, decomprima_imaginea
import cv2

#Variabile globale
default_cale_input = "input_video/test.mp4"
default_cale_output = "output_video/test_nou.mp4"


def proceseaza_video(cale_input=default_cale_input, cale_output=default_cale_output, max_frames=50):
    cap = cv2.VideoCapture(cale_input)
    
    if not cap.isOpened():
        print("Nu gasesc videoclipul!!")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Procesare video: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cale_output, fourcc, fps, (width, height))

    frame_count = 0
    total_original_bits = 0
    total_compressed_bits = 0

    while cap.isOpened():
        returned, frame = cap.read()
        
        if not returned or (max_frames and frame_count >= max_frames):
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rezultate = comprima_imaginea(frame_rgb)
        (im_comp, dicts, dim_orig, dim_pad, ql, qc) = rezultate
        
        bits_y, bits_cb, bits_cr = im_comp

        #biti comprimati
        frame_compressed_bits = len(bits_y) + len(bits_cb) + len(bits_cr)
        bits_dicts = 0

        #biti din dictionar

        for d in dicts:
            for simbol, cod in d.items():
                bits_dicts += 8 + len(cod)

        frame_compressed_bits = frame_compressed_bits + bits_dicts
        
        #biti necomprimati
        frame_original_bits = width * height * 3 * 8
        
        total_compressed_bits += frame_compressed_bits
        total_original_bits += frame_original_bits

        frame_reconstruit = decomprima_imaginea(im_comp, dicts, dim_orig, dim_pad, ql, qc)

        frame_out_bgr = cv2.cvtColor(frame_reconstruit, cv2.COLOR_RGB2BGR)
        
        out.write(frame_out_bgr)
        
        frame_count += 1
        print(f"Cadru {frame_count} procesat")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n" + "="*40)
    print(f"biti originali: {total_original_bits:,}")
    print(f"biti comprimati:  {total_compressed_bits:,}")
    print("="*40)

