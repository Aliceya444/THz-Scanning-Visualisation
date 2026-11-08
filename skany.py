import numpy as np
import matplotlib.pyplot as plt
import imageio
import re
import os

# === PATH CONFIGURATION ===

myFile='litera_4x4_pow3x_test_3.npy'           # Choose name of the measurement (one of many taken that day)


npy_path = 'C:\\Users\\user\\Desktop\\Pomiary\\09.10\\' + myFile             # Define path of the file above (used as saving path later) 
fileName = os.path.splitext(myFile)[0]

meta_path = npy_path.replace('.npy', '.meta')

# === DATA LOADING ===
data = np.load(npy_path)  # shape: (11, 64, 64)

# === TAKE VALUES FROM .META FILE ===
with open(meta_path, 'r') as f:
    meta_content = f.read()

start_z = float(re.search(r"Start Z:\s*([0-9.]+)", meta_content).group(1))
stop_z = float(re.search(r"Stop Z:\s*([0-9.]+)", meta_content).group(1))
step_z = float(re.search(r"Step Z:\s*([0-9.]+)", meta_content).group(1))

z_values = np.arange(start_z, stop_z + step_z, step_z)

# === OUTPUT FOLDERS ===
output_dir = os.path.join(os.path.dirname(npy_path), fileName)
gif_path = os.path.join(os.path.dirname(npy_path), "gify")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(gif_path, exist_ok=True)

# === SAVING 'Z' IMAGES WITH CAPTIONS ===
image_files = []

# Get Start/Stop for X and Y
start_x = float(re.search(r"Start X:\s*([0-9.]+)", meta_content).group(1))
stop_x = float(re.search(r"Stop X:\s*([0-9.]+)", meta_content).group(1))
start_y = float(re.search(r"Start Y:\s*([0-9.]+)", meta_content).group(1))
stop_y = float(re.search(r"Stop Y:\s*([0-9.]+)", meta_content).group(1))

# Preparation of the axis grid (mm)
extent = [start_x, stop_x, stop_y, start_y]  # WARNING: inverted Y (matplotlib and image)

# === GENERATE IMAGES WITH AXES AND LEGEND ===
image_files = []

for i, frame in enumerate(data):
    plt.figure(figsize=(6, 5))
    plt.imshow(frame, cmap='viridis', extent=extent, origin='upper', aspect='equal')
    plt.colorbar(label='Voltage [mV]')
    plt.title(f'Z = {z_values[i]:.1f} mm')
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    
    filename = os.path.join(output_dir, f'Z_{z_values[i]:.1f}mm.png')
    plt.savefig(filename)
    plt.close()
    
    image_files.append(filename)

# === CREATING GIF FILE ===
frames = [imageio.imread(img) for img in image_files]
gif_output = os.path.join(gif_path, fileName + '.gif')
imageio.mimsave(gif_output, frames, fps=2)

print()
print("\033[35mSaved images and GIF file:\n")            
print("Folder with images:\n", output_dir)
print("GIF file:\n", gif_output +'\n\n\033[0m')
