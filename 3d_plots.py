from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# Load and prepare the CSV
csv_path = "DATA_CHECKS_DURING_RUNS.csv"
df_raw = pd.read_csv(csv_path, header=None)
header = df_raw.iloc[1].astype(str).str.strip()
df = df_raw.iloc[2:].copy()
df.columns = header
df = df.reset_index(drop=True)

# Dynamic column names
run_col = [col for col in df.columns if "RUN" in str(col).upper()][0]
als_col = [col for col in df.columns if "X from TE" in str(col)][0]
speed_col = [col for col in df.columns if "ref speed" in str(col).lower()][0]

# Drag info
d_cols = ["D1", "D2", "D3", "D4"]
drag_pos = [391, 1000, 2274, 3472]

# Speeds to separate
speeds_to_plot = [4.0, 5.0, 6.0]

# Define a custom blue-to-red sequential colormap
blue_red = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

# Loop over each speed
for target_speed in speeds_to_plot:
    runs_data = defaultdict(lambda: {"X": [], "t": [], "drag": [], "temp": []})
    all_temp = []

    for idx, row in df.iterrows():
        try:
            # Extract and parse speed safely
            speed_str = str(row[speed_col])
            speed_match = re.search(r"(\d+(?:\.\d+)?)", speed_str)
            if not speed_match:
                continue
            speed = float(speed_match.group(1))
            if speed != target_speed:
                continue

            d_vals = [float(row[d]) for d in d_cols]
            als_string = str(row[als_col])
            match = re.findall(r"t=([\d.]+)\[mm\]\s*T=([\d.]+)\[Deg\]", als_string)
            if not match:
                continue
            t_val, T_val = map(float, match[0])

            run_id = str(row[run_col])

            runs_data[run_id]["X"].extend(drag_pos)
            runs_data[run_id]["drag"].extend(d_vals)
            runs_data[run_id]["t"].extend([t_val] * 4)
            runs_data[run_id]["temp"].extend([T_val] * 4)
            all_temp.extend([T_val] * 4)

        except Exception:
            continue  # Skip faulty rows

    # Skip if no data for this speed
    if len(all_temp) == 0:
        print(f"No valid data found for {target_speed} m/s.")
        continue

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    all_segments = []
    all_colors = []

    for run_id, data in runs_data.items():
        X = np.array(data["X"])
        t = np.array(data["t"])
        drag = np.array(data["drag"])
        temp = np.array(data["temp"])

        sorted_idx = np.argsort(X)
        X = X[sorted_idx]
        t = t[sorted_idx]
        drag = drag[sorted_idx]
        temp = temp[sorted_idx]

        points = np.array([X, t, drag]).T
        if len(points) < 2:
            continue
        segments = np.array([[points[i], points[i + 1]] for i in range(len(points) - 1)])
        all_segments.extend(segments)

        avg_temp = (temp[:-1] + temp[1:]) / 2
        all_colors.extend(avg_temp)

    # Normalize color scale for consistency
    norm = plt.Normalize(min(all_colors), max(all_colors))

    # Add colored lines
    lc = Line3DCollection(all_segments, cmap=blue_red, norm=norm)
    lc.set_array(np.array(all_colors))
    lc.set_linewidth(2)
    ax.add_collection3d(lc)

    # Scatter points with same colormap and normalization
    for run_id, data in runs_data.items():
        X = np.array(data["X"])
        t = np.array(data["t"])
        drag = np.array(data["drag"])
        temp = np.array(data["temp"])
        ax.scatter(X, t, drag, c=temp, cmap=blue_red, norm=norm, s=40, alpha=1)

    # Labels and colorbar
    ax.set_xlabel('Position X (mm)')
    ax.set_ylabel('Equivalent thickness t (mm)')
    ax.set_zlabel('Drag [-]')
    ax.set_title(f"Drag vs X and t (Colored by Temp) — {target_speed} m/s")

    cbar = plt.colorbar(lc, ax=ax, shrink=0.6)
    cbar.set_label('Temperature (°C)')

    plt.tight_layout()
    plt.savefig(f"drag_vs_X_t_temp_{int(target_speed)}ms_colored_lines.png", bbox_inches='tight')
    plt.show()
