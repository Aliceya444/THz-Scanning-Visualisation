#!/usr/bin/env python3
"""
find_best_focus_single.py

Analysis of a single .npy file + its paired .txt file
Simply change:
  - myFile -> the name of the .npy file
  - basePath -> the path to the folder containing the files
"""

import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# SELECT FILE FOR ANALYSIS

myFile = "swiatl_polaryz_skos_test_2.npy"

basePath = 'C:\\Users\\user\\Desktop\\Pomiary\\09.10\\'


npy_path = os.path.join(basePath, myFile)
txt_path = os.path.splitext(npy_path)[0] + ".txt"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ----------------------------
# '.txt' file parser
# ----------------------------
def parse_txt_info(txt_path):
    """Extracts a list of Z positions (float, in mm) from lines of the form '# z = ... mm'."""
    Z_positions = []
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().startswith("# z ="):
                parts = line.strip().split()
                try:
                    z_val = float(parts[3])  # np. "# z = 50.0 mm"
                    Z_positions.append(z_val)
                except (IndexError, ValueError):
                    continue
    return Z_positions

# ----------------------------
# Functions
# ----------------------------
def top_k_mean(arr2d, k_pixels=None, top_percent=None):
    flat = arr2d.ravel()
    if top_percent is not None:
        thr = np.percentile(flat, 100.0 - top_percent)
        sel = flat[flat >= thr]
        return float(sel.mean()) if sel.size > 0 else float(flat.max())
    else:
        k = max(1, int(k_pixels))
        if k >= flat.size:
            return float(flat.mean())
        kth = np.partition(flat, -k)[-k]
        sel = flat[flat >= kth]
        return float(sel.mean())

def rms_radius_max(frame):
    """RMS radius relative to the maximum."""
    arr = frame.astype(float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    ny, nx = arr.shape
    y0, x0 = np.unravel_index(np.argmax(arr), arr.shape)
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    R2 = (X - x0)**2 + (Y - y0)**2
    rms = np.sqrt((arr * R2).sum() / total)
    return float(rms)

def fwhm_1d(profile):
    maxv = np.max(profile)
    half = maxv / 2.0
    imax = np.argmax(profile)

    left, right = None, None
    for i in range(imax, -1, -1):
        if profile[i] <= half:
            if i < imax:
                t = (half - profile[i]) / (profile[i+1] - profile[i] + 1e-12)
                left = i + t
            else:
                left = i
            break
    for i in range(imax, len(profile)):
        if profile[i] <= half:
            if i > imax:
                t = (half - profile[i]) / (profile[i-1] - profile[i] + 1e-12)
                right = i - t
            else:
                right = i
            break
    if left is None or right is None:
        return np.nan
    return float(right - left)

# ----------------------------
# 'Z' stack analysis
# ----------------------------
def analyze_stack(stack, top_percent=1.0, energy_radius=3.0):
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    nz, ny, nx = stack.shape
    results = []
    for iz in range(nz):
        frame = stack[iz]
        topmean = top_k_mean(frame, top_percent=top_percent)
        rms = rms_radius_max(frame)

        # energia w promieniu (wokół maksimum)
        y0, x0 = np.unravel_index(np.argmax(frame), frame.shape)
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
        R = np.sqrt((X - x0)**2 + (Y - y0)**2)
        energy_in_r = float(frame[R <= energy_radius].sum())
        total_energy = frame.sum() if frame.sum() > 0 else 1.0
        energy_frac = energy_in_r / total_energy

        # FWHM_x, FWHM_y – przekroje przez maksimum
        profile_x = frame[y0, :]
        profile_y = frame[:, x0]
        fwhm_x = fwhm_1d(profile_x)
        fwhm_y = fwhm_1d(profile_y)

        results.append({
            "z_index": iz,
            "topmean": topmean,
            "rms_radius": rms,
            "energy_frac": energy_frac,
            "fwhm_x": fwhm_x,
            "fwhm_y": fwhm_y
        })
    return results

def normalize_metric(values):
    a = np.array(values, dtype=float)
    if np.nanmax(a) == np.nanmin(a):
        return np.zeros_like(a)
    return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))

# ----------------------------
# Main analysis of a single file
# ----------------------------
def process_file(npy_path, txt_path, top_percent=1.0, energy_radius=3.0, top_n=3):
    if not os.path.exists(npy_path):
        print(f"[ERROR] File missing {npy_path}")
        return
    if not os.path.exists(txt_path):
        print(f"[ERROR] File missing {txt_path}")
        return

    Z_positions = parse_txt_info(txt_path)
    stack = np.load(npy_path)
    results = analyze_stack(stack, top_percent=top_percent, energy_radius=energy_radius)

    ntop = normalize_metric([r["topmean"] for r in results])
    nenergy = normalize_metric([r["energy_frac"] for r in results])
    nrms = normalize_metric([r["rms_radius"] for r in results])
    nfwhm_x = normalize_metric([r["fwhm_x"] for r in results])
    nfwhm_y = normalize_metric([r["fwhm_y"] for r in results])

    for i, r in enumerate(results):
        r["score"] = ntop[i] + nenergy[i] - nrms[i] - 0.5*(nfwhm_x[i] + nfwhm_y[i])

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\nPlik: {os.path.basename(npy_path)} (Z slices: {len(results)})")
    for r in results_sorted[:top_n]:
        z_val = Z_positions[r["z_index"]] if r["z_index"] < len(Z_positions) else None
        print(f"  Z_index={r['z_index']:3d}, Z_val={z_val} mm | "
              f"score={r['score']:.3f} | topmean={r['topmean']:.3g} | "
              f"energy_frac={r['energy_frac']:.3g} | rms={r['rms_radius']:.3g} | "
              f"fwhm_x={r['fwhm_x']:.3g} | fwhm_y={r['fwhm_y']:.3g}")

    # CSV export
    csv_path = os.path.splitext(npy_path)[0] + "_wyniki.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Z_index", "Z_val_mm", "TopMean", "EnergyFrac", "RMS", "FWHM_X", "FWHM_Y", "Score"])
        for r in results:
            z_val = Z_positions[r["z_index"]] if r["z_index"] < len(Z_positions) else None
            writer.writerow([r["z_index"], z_val, r["topmean"],
                             r["energy_frac"], r["rms_radius"],
                             r["fwhm_x"], r["fwhm_y"], r["score"]])
    print(f"  -> Results saved to {csv_path}")

    # Charts
    Z_vals = [Z_positions[r["z_index"]] if r["z_index"] < len(Z_positions) else r["z_index"]
              for r in results]
    scores = [r["score"] for r in results]
    rms_vals = [r["rms_radius"] for r in results]
    fwhm_x_vals = [r["fwhm_x"] for r in results]
    fwhm_y_vals = [r["fwhm_y"] for r in results]

    fig, axs = plt.subplots(4, 1, figsize=(7, 12), sharex=True)
    axs[0].plot(Z_vals, scores, "-o")
    axs[0].set_ylabel("Score"); axs[0].grid(True)

    axs[1].plot(Z_vals, rms_vals, "-o", color="orange")
    axs[1].set_ylabel("RMS radius [px]"); axs[1].grid(True)

    axs[2].plot(Z_vals, fwhm_x_vals, "-o", color="green", label="FWHM_x")
    axs[2].plot(Z_vals, fwhm_y_vals, "-o", color="blue", label="FWHM_y")
    axs[2].set_ylabel("FWHM [px]"); axs[2].legend(); axs[2].grid(True)

    axs[3].plot(Z_vals, [r["topmean"] for r in results], "-o", color="purple")
    axs[3].set_xlabel("Z [mm]"); axs[3].set_ylabel("TopMean"); axs[3].grid(True)

    fig.suptitle(os.path.basename(npy_path))
    plt.tight_layout()
    png_path = os.path.splitext(npy_path)[0] + "_wykresy.png"
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  -> Charts saved to {png_path}")

# ----------------------------
# Launching
# ----------------------------
if __name__ == "__main__":
    process_file(npy_path, txt_path, top_percent=1.0, energy_radius=3.0, top_n=3)
