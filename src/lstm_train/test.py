import pandas as pd
import matplotlib.pyplot as plt

csv_path = "../../data/EEG_ET/EEG_ET_combined.csv"
df = pd.read_csv(csv_path)

# 1) Quick data inspection
print(df.head())          # first few rows
print(df.info())          # dtypes & non‐null counts
print(df.describe())      # basic statistics
print("Missing values:\n", df.isnull().sum())



# 2) Plot EEG and ET on separate subplots for clarity
eeg_ch = ["Pz", "F3"]
et_ch  = ["X", "Y"]
times  = df["time_s"].to_numpy()

fig, (ax_eeg, ax_et) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

# EEG subplot
for ch in eeg_ch:
    ax_eeg.plot(times, df[ch], label=ch)
ax_eeg.set_ylabel("EEG (μV)")
ax_eeg.set_title("EEG Channels")
ax_eeg.legend(loc="upper right")
ax_eeg.grid(True)

# Eye‐tracker subplot
for ch in et_ch:
    ax_et.plot(times, df[ch], label=ch)
ax_et.set_xlabel("Time (s)")
ax_et.set_ylabel("Eye‐Tracker (px)")
ax_et.set_title("Eye‐Tracker Coordinates")
ax_et.legend(loc="upper right")
ax_et.grid(True)

plt.tight_layout()
plt.show()
