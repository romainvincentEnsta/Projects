import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import re

# Set file name
run_id = 3037
filename = f'{run_id}_reduced_median_5_0.v'
basename = os.path.splitext(os.path.basename(filename))[0]

figures_dir = 'wall_vs_void_fractions'
os.makedirs(figures_dir, exist_ok=True)

plot_counter = 1
# Saving plots function
def save_current_plot_auto():
    global plot_counter
    filename = os.path.join(figures_dir, f"{basename}_plot{plot_counter}.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved : {filename}")
    plot_counter += 1

# Settings for pin selection
pin_start = 1
pin_end = 10
selected_pins = np.arange(pin_start, pin_end)

# time-resolved median filter 5 sampling reduced to 5 kHz
f = open(filename, mode='rb')
data = list(f.read())
sort_data_median3 = np.reshape(data, (90000, 32, 32))

Probe_1_median5 = sort_data_median3[:, :, 19]
Probe_2_median5 = sort_data_median3[:, :, 17]
Probe_3_median5 = sort_data_median3[:, :, 16]
Probe_4_median5 = sort_data_median3[:, :, 7]
Probe_5_median5 = sort_data_median3[:, :, 4]
Probe_6_median5 = sort_data_median3[:, :, 5]

i = 1  # ith pin on the probe, here we take the second pin because it gives the strongest signal
probes = {
    "Probe 1": Probe_1_median5[:, i],
    "Probe 2": Probe_2_median5[:, i],
    "Probe 3": Probe_3_median5[:, i],
    "Probe 4": Probe_4_median5[:, i],
    "Probe 5": Probe_5_median5[:, i],
    "Probe 6": Probe_6_median5[:, i],
}

t = np.arange(1, len(next(iter(probes.values()))) + 1)

# Steady OFF
alpha_steady_OFF = [[] for _ in range(6)]
for i in t:
    if i >= 0 and i <= 10000:
        alpha_steady_OFF[0].append(Probe_1_median5[i, :])
        alpha_steady_OFF[1].append(Probe_2_median5[i, :])
        alpha_steady_OFF[2].append(Probe_3_median5[i, :])
        alpha_steady_OFF[3].append(Probe_4_median5[i, :])
        alpha_steady_OFF[4].append(Probe_5_median5[i, :])
        alpha_steady_OFF[5].append(Probe_6_median5[i, :])

alpha_steady_OFF = [np.vstack([np.array(row) for row in alpha]) for alpha in alpha_steady_OFF]
alpha_avg_OFF_med5 = np.array([np.mean(alpha, axis=0) for alpha in alpha_steady_OFF])

# Steady ON
alpha_steady_ON = [[] for _ in range(6)]
for i in t:
    if i >= 65000 and i <= 75000:
        alpha_steady_ON[0].append(Probe_1_median5[i, :])
        alpha_steady_ON[1].append(Probe_2_median5[i, :])
        alpha_steady_ON[2].append(Probe_3_median5[i, :])
        alpha_steady_ON[3].append(Probe_4_median5[i, :])
        alpha_steady_ON[4].append(Probe_5_median5[i, :])
        alpha_steady_ON[5].append(Probe_6_median5[i, :])

alpha_steady_ON = [np.vstack([np.array(row) for row in alpha]) for alpha in alpha_steady_ON]
alpha_avg_ON_med5 = np.array([np.mean(alpha, axis=0) for alpha in alpha_steady_ON])

X_EC = [723, 1373, 1738, 2671, 3036, 3867]  # in mm
selected_probes = np.array([0, 1, 2, 3, 4, 5])
selected_X = np.array(X_EC)[selected_probes]

alpha_OFF_subset = alpha_avg_OFF_med5[np.ix_(selected_probes, selected_pins)]
alpha_ON_subset = alpha_avg_ON_med5[np.ix_(selected_probes, selected_pins)]

# Define logarithmic model
def log_fit(y, a, b):
    return a * np.log(y) + b

# Define wall-normal coordinates
y_EC_full = np.linspace(2, 64, 31)  # pins 1 to 31 (skip pin 0)
y_EC_selected = np.linspace(pin_start * 2, pin_end * 2, pin_end - pin_start)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Fit and plot for each probe (log scale on left, full on right)
for i in range(0, 2):
    # Full data for fit
    y_fit_all = y_EC_full
    signal_all = alpha_avg_ON_med5[i, 1:]

    # Perform fit on selected region
    valid_indices = (y_EC_selected >= 10) & (y_EC_selected <= 50)
    y_fit_sel = y_EC_selected[valid_indices]
    signal_sel = alpha_avg_ON_med5[i, pin_start:pin_end][valid_indices]

    try:
        popt_log, _ = curve_fit(log_fit, y_fit_sel, signal_sel)

        # Plot left: fit in log scale with selected region
        ax1.plot(signal_sel, y_fit_sel, 'o', label=f'Probe {i+1} data')
        ax1.plot(log_fit(y_EC_selected, *popt_log), y_EC_selected, '--', label=f'Probe {i+1} Fit')

        # Compute R² for log fit
        y_pred_sel = log_fit(y_fit_sel, *popt_log)
        ss_res = np.sum((signal_sel - y_pred_sel) ** 2)
        ss_tot = np.sum((signal_sel - np.mean(signal_sel)) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax1.text(0.95, 0.05 + 0.05*i, f"Probe {i+1} $R^2$ = {r2:.3f}", transform=ax1.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

        # Plot right: raw points only
        ax2.plot(signal_all, y_EC_full, 'o', label=f'Probe {i+1} data')

    except RuntimeError:
        print(f"Fit failed for Probe {i+1}")

ax1.set_xscale('log')
ax1.set_xlabel("Void fraction α [%] (log scale)")
ax1.set_ylabel("y_EC")
ax1.set_title(f"Log Fit on Selected Pins [{pin_start,pin_end}]")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Void fraction α [%]")
ax2.set_title("Raw Data on Full Pin Range")
ax2.grid(True)
ax2.legend()

plt.show()
