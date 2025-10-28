import os
import numpy as np
import matplotlib.pyplot as plt

output_imagini = os.path.join("imagini")
os.makedirs(output_imagini, exist_ok=True)

def save_folder(nume_file):
    return os.path.join(output_imagini,nume_file)