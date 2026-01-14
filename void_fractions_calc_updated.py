
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

# -------------------------
# User parameters
# -------------------------
run_id = 4027
filename = f'{run_id}_reduced_median_5_0.v'
basename = os.path.splitext(os.path.basename(filename))[0]

# Directory for figures
figures_dir = 'wall_vs_void_fractions'
os.makedirs(figures_dir, exist_ok=True)

plot_counter = 1
def save_current_plot_auto():
    global plot_counter
    fname = os.path.join(figures_dir, f"{basename}_plot{plot_counter}.png")
    plt.savefig(fname, bbox_inches='tight')
    print(f"Plot saved: {fname}")
    plot_counter += 1

# -------------------------
# Load binary data and adapt to file length
# -------------------------
with open(filename, mode='rb') as f:
    data = f.read()

# Expect 32x32 bytes per frame
BYTES_PER_FRAME = 32 * 32
total_bytes = len(data)
n_frames_exact = total_bytes // BYTES_PER_FRAME
remainder = total_bytes % BYTES_PER_FRAME

if n_frames_exact == 0:
    raise ValueError(f"No complete frames found in {filename}. File too small?")

if remainder != 0:
    # Truncate any partial frame at the end for safety
    print(f"Warning: file has {remainder} extra bytes (partial frame). Truncating to {n_frames_exact} full frames.")

# Convert to numpy and reshape
arr = np.frombuffer(data[:n_frames_exact * BYTES_PER_FRAME], dtype=np.uint8)
sort_data_median3 = arr.reshape(n_frames_exact, 32, 32)

# -------------------------
# Probe extraction (time, pin index)
# -------------------------
# Choose second pin (0-based index = 1)
pin_index = 1

Probe_1_median5 = sort_data_median3[:, :, 19]
Probe_2_median5 = sort_data_median3[:, :, 17]
Probe_3_median5 = sort_data_median3[:, :, 16]
Probe_4_median5 = sort_data_median3[:, :, 7]
Probe_5_median5 = sort_data_median3[:, :, 4]
Probe_6_median5 = sort_data_median3[:, :, 5]

probes = {
    "Probe 1": Probe_1_median5[:, pin_index],
    "Probe 2": Probe_2_median5[:, pin_index],
    "Probe 3": Probe_3_median5[:, pin_index],
    "Probe 4": Probe_4_median5[:, pin_index],
    "Probe 5": Probe_5_median5[:, pin_index],
    "Probe 6": Probe_6_median5[:, pin_index],
}

# Time indices (adapts automatically to file length)
t = np.arange(n_frames_exact)

# -------------------------
# Window selection
# -------------------------
# OFF window: first up to 10,000 samples (or fewer if file is shorter)
off_end = min(10_000, n_frames_exact)
off_slice = slice(0, off_end)

# ON window: ALWAYS last 15,000 points (or the entire file if shorter)
on_start = max(0, n_frames_exact - 15_000)
on_slice = slice(on_start, n_frames_exact)

print(f"Detected frames: {n_frames_exact}")
print(f"OFF window: indices [{off_slice.start}:{off_slice.stop}] (length={off_end})")
print(f"ON  window: indices [{on_slice.start}:{on_slice.stop}] (length={on_slice.stop - on_slice.start})")

# -------------------------
# Aggregate per-probe means across time for OFF/ON windows
# (Resulting arrays have shape (6, 32) for pins)
# -------------------------
alpha_avg_OFF_med5 = np.zeros((6, 32))
alpha_avg_ON_med5  = np.zeros((6, 32))

# OFF
alpha_avg_OFF_med5[0, :] = Probe_1_median5[off_slice, :].mean(axis=0)
alpha_avg_OFF_med5[1, :] = Probe_2_median5[off_slice, :].mean(axis=0)
alpha_avg_OFF_med5[2, :] = Probe_3_median5[off_slice, :].mean(axis=0)
alpha_avg_OFF_med5[3, :] = Probe_4_median5[off_slice, :].mean(axis=0)
alpha_avg_OFF_med5[4, :] = Probe_5_median5[off_slice, :].mean(axis=0)
alpha_avg_OFF_med5[5, :] = Probe_6_median5[off_slice, :].mean(axis=0)

# ON
alpha_avg_ON_med5[0, :]  = Probe_1_median5[on_slice, :].mean(axis=0)
alpha_avg_ON_med5[1, :]  = Probe_2_median5[on_slice, :].mean(axis=0)
alpha_avg_ON_med5[2, :]  = Probe_3_median5[on_slice, :].mean(axis=0)
alpha_avg_ON_med5[3, :]  = Probe_4_median5[on_slice, :].mean(axis=0)
alpha_avg_ON_med5[4, :]  = Probe_5_median5[on_slice, :].mean(axis=0)
alpha_avg_ON_med5[5, :]  = Probe_6_median5[on_slice, :].mean(axis=0)

# -------------------------
# Plot mean void fractions at selected pins across probes
# -------------------------
X_EC = [723, 1373, 1738, 2671, 3036, 3867]  # in mm
selected_probes = np.array([0, 1, 2, 3, 4, 5])
selected_pins = np.array([pin_index])  # second pin
selected_X = np.array(X_EC)[selected_probes]

alpha_OFF_subset = alpha_avg_OFF_med5[np.ix_(selected_probes, selected_pins)]
alpha_ON_subset  = alpha_avg_ON_med5[np.ix_(selected_probes, selected_pins)]

plt.figure(figsize=(10, 6))
for idx, pin in enumerate(selected_pins):
    plt.plot(selected_X, alpha_OFF_subset[:, idx], '-o', label=f'Pin {pin} OFF')
    plt.plot(selected_X, alpha_ON_subset[:, idx], '--x', label=f'Pin {pin} ON')

plt.xlabel('Position X (mm)')
plt.ylabel('Void fraction α [%]')
plt.title('Mean void fractions for all probes for different pin locations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)

# -------------------------
# Load CSV with run metadata (unchanged from your script)
# -------------------------
csv_path = "Sheet5.csv"
df_raw = pd.read_csv(csv_path, header=None)

header = df_raw.iloc[1].astype(str).str.strip()
df = df_raw.iloc[2:].copy()
df.columns = header
df = df.reset_index(drop=True)

run_col = [col for col in df.columns if "RUN" in str(col).upper()]
if not run_col:
    raise ValueError("Couldn't find a column containing 'RUN' in the header.")
run_col = run_col[0]

df[run_col] = pd.to_numeric(df[run_col], errors="coerce")
row = df[df[run_col] == run_id]
if row.empty:
    raise ValueError(f"Run {run_id} not found in the CSV.")

D1 = float(row["D1"].values[0])
D2 = float(row["D2"].values[0])
D3 = float(row["D3"].values[0])
D4 = float(row["D4"].values[0])

D_values = [D1, D2, D3, D4]
D_labels = ["D1", "D2", "D3", "D4"]
D_x_positions = [391, 1000, 2274, 3472]

# -------------------------
# Void fraction + drag overlay (kept from your script)
# -------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))

for idx, pin in enumerate(selected_pins):
    ax1.plot(selected_X, alpha_OFF_subset[:, idx], '-o', label=f'Pin {pin} OFF')
    ax1.plot(selected_X, alpha_ON_subset[:, idx], '--x', label=f'Pin {pin} ON')

ax1.set_xlabel('Position X (mm)')
ax1.set_ylabel('Void fraction α [%]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(D_x_positions, D_values, 's-', color='tab:red', label='Drag (D1, D2, D3, D4)', markersize=8)

for x, y, label in zip(D_x_positions, D_values, D_labels):
    ax2.text(x, y + 0.01, label, ha='center', va='bottom', fontsize=9, color='tab:red')

ax2.set_ylabel('Drag [-]', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(-0.1, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title(f'Mean void fractions and Drag values (Run {run_id})')
plt.tight_layout()
plt.show(block=False)

# -------------------------
# Integrated alpha vs X with drag overlay (kept, uses updated means)
# -------------------------
delta_y = 2  # spacing in mm
integral_alpha_avg_on = np.sum(alpha_avg_ON_med5 * delta_y, axis=1)
integral_alpha_avg_off = np.sum(alpha_avg_OFF_med5 * delta_y, axis=1)

X_EC = [723, 1373, 1738, 2671, 3036, 3867]

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(X_EC, integral_alpha_avg_on, '-o', label='∫α dy (ON)', color='tab:blue')
ax1.plot(X_EC, integral_alpha_avg_off, '--x', label='∫α dy (OFF)', color='tab:orange')

ax1.set_xlabel("Probe Position X (mm)")
ax1.set_ylabel("Integrated Void Fraction ∫α dy [mm]", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx()
drag_x = [391, 1000, 2274, 3472]
drag_y = [D1, D2, D3, D4]
drag_labels = ["D1", "D2", "D3", "D4"]
ax2.plot(drag_x, drag_y, 's--', color='tab:red', label='Drag D')
for x, y, label in zip(drag_x, drag_y, drag_labels):
    ax2.text(x, y + 0.01, label, ha='center', va='bottom', color='tab:red')

ax2.set_ylabel("Drag [-]", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title(f'Integrated Void Fraction (ON/OFF) vs Drag (Run {run_id})')
plt.tight_layout()
plt.show(block=False)

# -------------------------
# Interpolation at drag positions and correlation (unchanged)
# -------------------------
dy = 2  # mm spacing
integral_alpha_percent = np.sum(alpha_avg_ON_med5 * dy, axis=1) * 100

probe_X = np.array([723, 1373, 1738, 2671, 3036, 3867])
drag_X = np.array([391, 1000, 2274, 3472])
drag_Y = np.array([D1, D2, D3, D4])

interp_func = interp1d(probe_X, integral_alpha_percent, kind='linear', fill_value="extrapolate")
alpha_at_drag = interp_func(drag_X)

pearson_corr, _ = pearsonr(alpha_at_drag, drag_Y)
spearman_corr, _ = spearmanr(alpha_at_drag, drag_Y)

print("Interpolated ∫α(y)dy at drag positions:", alpha_at_drag)
print("Drag values:", drag_Y)
print(f"Pearson correlation: {pearson_corr:.3f}")
print(f"Spearman correlation: {spearman_corr:.3f}")

plt.figure()
plt.plot(drag_X, alpha_at_drag, marker='o')
plt.xlabel('X (mm)')
plt.ylabel('Interpolated ∫α dy [%·mm]')
plt.grid(True)
plt.show(block=False)

# -------------------------
# CSV export: ALL void fraction values for each probe, for the SECOND PIN
# We export both OFF and ON windows for completeness.
# -------------------------
on_len = on_slice.stop - on_slice.start
off_len = off_slice.stop - off_slice.start

# Build DataFrames (index = sample number within the window)
on_df = pd.DataFrame({
    "Probe 1": Probe_1_median5[on_slice, pin_index],
    "Probe 2": Probe_2_median5[on_slice, pin_index],
    "Probe 3": Probe_3_median5[on_slice, pin_index],
    "Probe 4": Probe_4_median5[on_slice, pin_index],
    "Probe 5": Probe_5_median5[on_slice, pin_index],
    "Probe 6": Probe_6_median5[on_slice, pin_index],
})
on_df.index.name = "sample_index"

off_df = pd.DataFrame({
    "Probe 1": Probe_1_median5[off_slice, pin_index],
    "Probe 2": Probe_2_median5[off_slice, pin_index],
    "Probe 3": Probe_3_median5[off_slice, pin_index],
    "Probe 4": Probe_4_median5[off_slice, pin_index],
    "Probe 5": Probe_5_median5[off_slice, pin_index],
    "Probe 6": Probe_6_median5[off_slice, pin_index],
})
off_df.index.name = "sample_index"

on_csv  = f"void_fractions_ON_pin2_RUN{run_id}.csv"
off_csv = f"void_fractions_OFF_pin2_RUN{run_id}.csv"
on_df.to_csv(on_csv)
off_df.to_csv(off_csv)

print(f"Saved ON  window CSV ({on_len} rows): {on_csv}")
print(f"Saved OFF window CSV ({off_len} rows): {off_csv}")

# -------------------------
# Summary: average of second pin across probes (ON and OFF)
# -------------------------
avg_void_fraction_per_probe_ON = alpha_avg_ON_med5[:, pin_index]
avg_void_fraction_per_probe_OFF = alpha_avg_OFF_med5[:, pin_index]

avg_all_ON = np.mean(avg_void_fraction_per_probe_ON)
avg_all_OFF = np.mean(avg_void_fraction_per_probe_OFF)

print("\\n--- Average Void Fraction for Second Pin (index=1) ---")
for probe_idx, (v_on, v_off) in enumerate(zip(avg_void_fraction_per_probe_ON, avg_void_fraction_per_probe_OFF), start=1):
    print(f"Probe {probe_idx} ON:  {v_on:.3f} %   OFF:  {v_off:.3f} %")

print(f"\\nMean over all probes (ON):  {avg_all_ON:.3f} %")
print(f"Mean over all probes (OFF): {avg_all_OFF:.3f} %")
