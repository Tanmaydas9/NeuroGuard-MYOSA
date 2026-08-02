"""
NeuroGuard session analyzer.
Usage:  python analyze_neuroguard.py neuroguard_log.csv  [session_id]

Outputs PNG plots + a printed summary covering:
  - state distribution
  - emergency events and intervention outcomes
  - physiological trends (HR, SpO2, HRV)
  - BMP180 barometric pressure, temperature, and altitude drops
  - APDS9960 proximity buildup and ambient light context
  - motion signature analysis (jerk, free-fall, impact)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "neuroguard_log.csv"
df = pd.read_csv(CSV)

if len(sys.argv) > 2:
    df = df[df["session"] == int(sys.argv[2])]
else:
    # default: most recent session in the log
    df = df[df["session"] == df["session"].iloc[-1]]

df = df.reset_index(drop=True)
df["t_s"] = df["t_ms"] / 1000.0

# ============================================================
# Summary metrics
# ============================================================
total_s   = df["t_s"].iloc[-1] - df["t_s"].iloc[0]
state_pct = df["state"].value_counts(normalize=True) * 100

# Emergency events
EMERGENCY_CLASSES = ["SLEEPWALK", "SEIZURE", "FREEZING", "FALL", "SYNCOPE"]

# Intervention outcomes
inflations = int((df["mask"] > 0).sum())
cancels    = int((df["cancel"] == 1).sum())

# Physiological aggregates
mean_hr   = df["hr"][df["hr"] > 30].mean()     if (df["hr"] > 30).any()     else 0
mean_spo2 = df["spo2"][df["spo2"] > 80].mean() if (df["spo2"] > 80).any() else 0
mean_hrv  = df["hrv"].mean()
min_spo2  = df["spo2"][df["spo2"] > 80].min()  if (df["spo2"] > 80).any() else 0

# BMP180 environmental aggregates
mean_press   = df["press"].mean() if "press" in df.columns else 0
mean_temp    = df["temp"].mean()  if "temp"  in df.columns else 0
max_alt_drop = (df["alt"].iloc[0] - df["alt"].min()) if "alt" in df.columns else 0
min_alt      = df["alt"].min() if "alt" in df.columns else 0

# APDS9960 proximity + ambient light aggregates
max_prox = df["prox"].max() if "prox" in df.columns else 0
ambient_col = "lux" if "lux" in df.columns else ("ambient" if "ambient" in df.columns else None)
mean_ambient = df[ambient_col].mean() if ambient_col else 0
nocturnal_frac = (df[ambient_col] < 40).mean() if ambient_col else 0

# Motion signature
mean_jerk        = df["jerk"].mean()
peak_accel       = df["accelMag"].max()
free_fall_events = int((df["accelMag"] < 0.4).sum())
stability_idx    = 1000.0 / (mean_jerk + 1)

print("=" * 60)
print(f"Session: {df['session'].iloc[0]}   Duration: {total_s:.1f}s   Samples: {len(df)}")
print("-" * 60)
print("State distribution (%):")
print(state_pct.round(1).to_string())
print("-" * 60)
print(f"Total inflations fired:     {inflations}")
print(f"User cancellations:         {cancels}")
print(f"Free-fall detections:       {free_fall_events}")
print(f"Peak accel magnitude (g):   {peak_accel:.2f}")
print(f"Mean jerk:                  {mean_jerk:.2f}")
print(f"Motion stability index:     {stability_idx:.2f}")
print("-" * 60)
print(f"Mean heart rate (bpm):      {mean_hr:.1f}")
print(f"Mean SpO2 (%):              {mean_spo2:.1f}")
print(f"Min SpO2 (%):               {min_spo2:.1f}")
print(f"Mean HRV RMSSD (ms):        {mean_hrv:.1f}")
print("-" * 60)
print("BMP180 environmental:")
print(f"  Mean pressure (hPa):      {mean_press:.2f}")
print(f"  Mean temperature (C):     {mean_temp:.2f}")
print(f"  Max altitude drop (m):    {max_alt_drop:.3f}")
print(f"  Min altitude delta (m):   {min_alt:.3f}")
print("-" * 60)
print("APDS9960 proximity + light:")
print(f"  Peak proximity reading:   {max_prox}")
print(f"  Mean ambient light:       {mean_ambient:.1f}")
print(f"  Nocturnal time fraction:  {nocturnal_frac*100:.1f}%")
print("=" * 60)

# ============================================================
# Plots
# ============================================================
fig, ax = plt.subplots(7, 1, figsize=(12, 16), sharex=True)

# Row 0: accel magnitude with fall thresholds
ax[0].plot(df["t_s"], df["accelMag"], label="Accel magnitude (g)", lw=1)
ax[0].axhline(0.4, color="red",    linestyle="--", alpha=0.5, label="free-fall")
ax[0].axhline(2.5, color="orange", linestyle="--", alpha=0.5, label="impact")
ax[0].set_ylabel("|a| (g)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# Row 1: gyro magnitude
ax[1].plot(df["t_s"], df["gyroMag"], color="purple", label="Gyro magnitude", lw=1)
ax[1].set_ylabel("|w|")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

# Row 2: jerk
ax[2].plot(df["t_s"], df["jerk"], color="orange", label="Jerk", lw=1)
ax[2].set_ylabel("Jerk")
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

# Row 3: HR + SpO2 (twin-axis)
ax[3].plot(df["t_s"], df["hr"], color="crimson", label="HR (bpm)", lw=1)
ax3b = ax[3].twinx()
ax3b.plot(df["t_s"], df["spo2"], color="steelblue", label="SpO2 (%)",
          lw=1, alpha=0.7)
ax[3].set_ylabel("HR", color="crimson")
ax3b.set_ylabel("SpO2", color="steelblue")
ax[3].grid(alpha=0.3)

# Row 4: BMP180 pressure + altitude (twin-axis)
if "press" in df.columns:
    ax[4].plot(df["t_s"], df["press"], color="darkgreen",
               label="Pressure (hPa)", lw=1)
    ax[4].set_ylabel("Pressure (hPa)", color="darkgreen")
    if "alt" in df.columns:
        ax4b = ax[4].twinx()
        ax4b.plot(df["t_s"], df["alt"], color="brown",
                  label="Alt delta (m)", lw=1, alpha=0.7)
        ax4b.axhline(-0.4, color="red", linestyle="--", alpha=0.5,
                     label="fall drop threshold")
        ax4b.set_ylabel("Alt delta (m)", color="brown")
    ax[4].legend(fontsize=8); ax[4].grid(alpha=0.3)
else:
    ax[4].text(0.5, 0.5, "BMP180 columns not present in log",
               ha="center", va="center", transform=ax[4].transAxes)

# Row 5: APDS9960 proximity + ambient light (twin-axis)
if "prox" in df.columns:
    ax[5].plot(df["t_s"], df["prox"], color="magenta",
               label="Proximity", lw=1)
    ax[5].axhline(60, color="red", linestyle="--", alpha=0.5,
                  label="sleepwalk warn")
    ax[5].set_ylabel("Proximity", color="magenta")
    if ambient_col:
        ax5b = ax[5].twinx()
        ax5b.plot(df["t_s"], df[ambient_col], color="gold",
                  label="Ambient", lw=1, alpha=0.7)
        ax5b.axhline(40, color="darkblue", linestyle="--", alpha=0.4,
                     label="nocturnal threshold")
        ax5b.set_ylabel("Ambient", color="gold")
    ax[5].legend(fontsize=8); ax[5].grid(alpha=0.3)
else:
    ax[5].text(0.5, 0.5, "APDS9960 columns not present in log",
               ha="center", va="center", transform=ax[5].transAxes)

# Row 6: State timeline
state_codes = {
    "IDLE":      0, "WALKING":   1, "RESTING":   2, "ACTIVITY":  3,
    "SLEEPWALK": 4, "FREEZING":  5, "SEIZURE":   6, "FALL":      7,
    "SYNCOPE":   8, "CANCEL?":   9, "INFLATE":  10, "COOLDN":   11,
}
ax[6].plot(df["t_s"], df["state"].map(state_codes).fillna(-1),
           drawstyle="steps-post", color="darkgreen")
ax[6].set_yticks(list(state_codes.values()))
ax[6].set_yticklabels(list(state_codes.keys()), fontsize=8)
ax[6].set_xlabel("Time (s)")
ax[6].grid(alpha=0.3)

# Highlight inflation events with vertical spans across every subplot
for _, row in df[df["mask"] > 0].iterrows():
    for a in ax:
        a.axvspan(row["t_s"] - 0.05, row["t_s"] + 0.05,
                  color="red", alpha=0.15)

plt.suptitle(f"NeuroGuard Session {df['session'].iloc[0]}")
plt.tight_layout()
out = f"session_{df['session'].iloc[0]}.png"
plt.savefig(out, dpi=140)
print(f"Saved plot -> {out}")
