"""
NeuroGuard synthetic dataset generator.
Produces physiologically plausible 20Hz sensor data for 8 neurological classes.
Use as bootstrap data for CNN+BiLSTM+Attention training.

Signals per row:
  Motion (MPU6050 @ 0x69):  ax, ay, az, gx, gy, gz
  Physio (MAX30102):        hr, spo2, hrv
  Environmental (BMP180):   pressure_hpa, temp_c, altitude_m
  Proximity+Light (APDS9960): prox, ambient, red, green, blue

Classes:
  0 normal_walking      - baseline locomotion (daytime, bright ambient)
  1 resting             - stationary posture
  2 daily_activity      - varied normal movements
  3 sleepwalking        - slow gait, low HR, high HRV, DARK ambient, rising prox
  4 epileptic_seizure   - 4-8 Hz high-amplitude convulsions + HR spike
  5 freezing_of_gait    - Parkinsonian low-amp 4-8 Hz tremor + stationary base
  6 fall                - free-fall + impact + stillness + barometric altitude drop
  7 syncope             - HR crash + gradual postural collapse + slight altitude drop
"""
import numpy as np
import pandas as pd

np.random.seed(42)
FS = 20  # Hz, matches firmware sampling rate

CLASSES = {
    "normal_walking":     {"weight": 0.18, "duration_s": (8, 15)},
    "resting":            {"weight": 0.18, "duration_s": (10, 20)},
    "daily_activity":     {"weight": 0.15, "duration_s": (5, 12)},
    "sleepwalking":       {"weight": 0.10, "duration_s": (8, 15)},
    "epileptic_seizure":  {"weight": 0.10, "duration_s": (4, 10)},
    "freezing_of_gait":   {"weight": 0.10, "duration_s": (3, 8)},
    "fall":               {"weight": 0.09, "duration_s": (2, 4)},
    "syncope":            {"weight": 0.10, "duration_s": (3, 6)},
}

# Baseline environmental parameters (indoor room, sea-level)
BASE_PRESSURE_HPA = 1013.25
BASE_ALT_M        = 0.0
BASE_TEMP_C       = 25.0


def gen_segment(label, n_samples):
    """Returns dict of feature arrays for one labeled segment."""
    t = np.arange(n_samples) / FS
    seg = {}

    # Gravity vector defaults (accel in g, gyro in deg/s)
    ax_b, ay_b, az_b = 0.0, 0.0, 1.0
    gx_b = gy_b = gz_b = 0.0

    # Default environmental context: bright daytime, no proximity obstacle
    press = BASE_PRESSURE_HPA + np.random.normal(0, 0.3, n_samples)
    temp  = BASE_TEMP_C       + np.random.normal(0, 0.05, n_samples)
    alt   = BASE_ALT_M        + np.random.normal(0, 0.05, n_samples)
    prox  = np.random.randint(0, 15, n_samples)          # low ambient proximity
    ambient = np.random.randint(400, 800, n_samples)     # bright daytime
    red   = (ambient * 0.55).astype(int) + np.random.randint(-30, 30, n_samples)
    green = (ambient * 0.60).astype(int) + np.random.randint(-30, 30, n_samples)
    blue  = (ambient * 0.45).astype(int) + np.random.randint(-30, 30, n_samples)

    if label == "normal_walking":
        f = np.random.uniform(1.6, 2.2)
        phase = np.random.uniform(0, 2 * np.pi)
        ax = ax_b + 0.15 * np.sin(2 * np.pi * f * t + phase)       + np.random.normal(0, 0.04, n_samples)
        ay = ay_b + 0.12 * np.cos(2 * np.pi * f * t + phase)       + np.random.normal(0, 0.04, n_samples)
        az = az_b + 0.25 * np.sin(2 * np.pi * f * t + phase + 1.2) + np.random.normal(0, 0.05, n_samples)
        gx = 40 * np.sin(2 * np.pi * f * t + phase) + np.random.normal(0, 3, n_samples)
        gy = 25 * np.cos(2 * np.pi * f * t + phase) + np.random.normal(0, 3, n_samples)
        gz = 10 * np.sin(2 * np.pi * f * t + phase + 0.5) + np.random.normal(0, 3, n_samples)
        hr   = np.random.uniform(85, 110, n_samples) + np.random.normal(0, 1.5, n_samples)
        spo2 = np.random.uniform(96, 99, n_samples)  + np.random.normal(0, 0.3, n_samples)
        hrv  = np.random.uniform(25, 55, n_samples)  + np.random.normal(0, 3, n_samples)
        # Walking modulates barometric pressure very slightly (step-to-step)
        press += 0.05 * np.sin(2 * np.pi * f * t)
        alt   += 0.02 * np.sin(2 * np.pi * f * t)

    elif label == "resting":
        ax = ax_b + np.random.normal(0, 0.02, n_samples)
        ay = ay_b + np.random.normal(0, 0.02, n_samples)
        az = az_b + np.random.normal(0, 0.02, n_samples)
        gx = np.random.normal(0, 1.0, n_samples)
        gy = np.random.normal(0, 1.0, n_samples)
        gz = np.random.normal(0, 1.0, n_samples)
        hr   = np.random.uniform(58, 78, n_samples) + np.random.normal(0, 1.0, n_samples)
        spo2 = np.random.uniform(97, 99, n_samples) + np.random.normal(0, 0.2, n_samples)
        hrv  = np.random.uniform(40, 90, n_samples) + np.random.normal(0, 4, n_samples)

    elif label == "daily_activity":
        f = np.random.uniform(0.5, 3.0)
        amp = np.random.uniform(0.3, 0.8)
        ax = ax_b + amp * np.sin(2 * np.pi * f * t)         + np.random.normal(0, 0.1, n_samples)
        ay = ay_b + amp * np.cos(2 * np.pi * f * t * 0.7)   + np.random.normal(0, 0.1, n_samples)
        az = az_b + 0.3 * np.sin(2 * np.pi * f * t)         + np.random.normal(0, 0.1, n_samples)
        gx = 60 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 6, n_samples)
        gy = 45 * np.cos(2 * np.pi * f * t) + np.random.normal(0, 6, n_samples)
        gz = 20 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 6, n_samples)
        hr   = np.random.uniform(90, 130, n_samples) + np.random.normal(0, 3, n_samples)
        spo2 = np.random.uniform(95, 99, n_samples)  + np.random.normal(0, 0.4, n_samples)
        hrv  = np.random.uniform(20, 50, n_samples)  + np.random.normal(0, 4, n_samples)
        # Occasional altitude changes (stairs, sitting down/up)
        if np.random.random() < 0.3:
            alt += np.linspace(0, np.random.uniform(-0.5, 0.5), n_samples)

    elif label == "sleepwalking":
        # Slower gait, low arousal, HIGH HRV, DARK ambient, PROX rises (obstacle)
        f = np.random.uniform(0.8, 1.3)
        ax = ax_b + 0.08 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 0.03, n_samples)
        ay = ay_b + 0.06 * np.cos(2 * np.pi * f * t) + np.random.normal(0, 0.03, n_samples)
        az = az_b + 0.12 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 0.03, n_samples)
        gx = 20 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 2, n_samples)
        gy = 15 * np.cos(2 * np.pi * f * t) + np.random.normal(0, 2, n_samples)
        gz = 8  * np.sin(2 * np.pi * f * t) + np.random.normal(0, 2, n_samples)
        hr   = np.random.uniform(55, 75, n_samples)  + np.random.normal(0, 1.0, n_samples)
        spo2 = np.random.uniform(95, 98, n_samples)  + np.random.normal(0, 0.3, n_samples)
        hrv  = np.random.uniform(60, 110, n_samples) + np.random.normal(0, 6, n_samples)
        # DARK bedroom (APDS9960 ambient stays low)
        ambient = np.random.randint(5, 40, n_samples)
        red     = (ambient * 0.4).astype(int) + np.random.randint(-2, 2, n_samples)
        green   = (ambient * 0.5).astype(int) + np.random.randint(-2, 2, n_samples)
        blue    = (ambient * 0.6).astype(int) + np.random.randint(-2, 2, n_samples)
        # Proximity increases as wearer approaches obstacle
        prox    = np.clip(np.linspace(10, 180, n_samples).astype(int) +
                          np.random.randint(-8, 8, n_samples), 0, 255)
        temp   += np.random.normal(-2.0, 0.1, n_samples)  # cooler at night

    elif label == "epileptic_seizure":
        f = np.random.uniform(4.0, 8.0)
        amp = np.random.uniform(1.5, 3.0)
        ax = ax_b + amp * np.sin(2 * np.pi * f * t)         + np.random.normal(0, 0.3, n_samples)
        ay = ay_b + amp * np.cos(2 * np.pi * f * t)         + np.random.normal(0, 0.3, n_samples)
        az = az_b + amp * np.sin(2 * np.pi * f * t + 0.5)   + np.random.normal(0, 0.3, n_samples)
        gx = 300 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 20, n_samples)
        gy = 250 * np.cos(2 * np.pi * f * t) + np.random.normal(0, 20, n_samples)
        gz = 180 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 20, n_samples)
        hr   = np.random.uniform(130, 180, n_samples) + np.random.normal(0, 5, n_samples)
        spo2 = np.random.uniform(88, 94, n_samples)   + np.random.normal(0, 1.0, n_samples)
        hrv  = np.random.uniform(5, 20, n_samples)    + np.random.normal(0, 2, n_samples)
        # Barometric pressure shakes with convulsions
        press += 0.2 * np.random.normal(0, 1, n_samples)

    elif label == "freezing_of_gait":
        f = np.random.uniform(4.5, 7.5)
        amp = np.random.uniform(0.05, 0.15)
        ax = ax_b + amp * np.sin(2 * np.pi * f * t) + np.random.normal(0, 0.02, n_samples)
        ay = ay_b + amp * np.cos(2 * np.pi * f * t) + np.random.normal(0, 0.02, n_samples)
        az = az_b + amp * np.sin(2 * np.pi * f * t) + np.random.normal(0, 0.02, n_samples)
        gx = 8 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 1.5, n_samples)
        gy = 6 * np.cos(2 * np.pi * f * t) + np.random.normal(0, 1.5, n_samples)
        gz = 4 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 1.5, n_samples)
        hr   = np.random.uniform(80, 110, n_samples) + np.random.normal(0, 2, n_samples)
        spo2 = np.random.uniform(96, 99, n_samples)  + np.random.normal(0, 0.3, n_samples)
        hrv  = np.random.uniform(15, 35, n_samples)  + np.random.normal(0, 3, n_samples)

    elif label == "fall":
        ax = np.zeros(n_samples); ay = np.zeros(n_samples); az = np.zeros(n_samples)
        gx = np.zeros(n_samples); gy = np.zeros(n_samples); gz = np.zeros(n_samples)
        seg_a = min(int(0.5 * FS), n_samples // 3)
        seg_b = min(int(0.35 * FS), n_samples // 3)
        seg_c = min(int(0.10 * FS), 3)
        tt = np.linspace(0, 0.5, seg_a)
        ax[:seg_a] = 0.15 * np.sin(2 * np.pi * 2 * tt)
        ay[:seg_a] = 0.12 * np.cos(2 * np.pi * 2 * tt)
        az[:seg_a] = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * tt)
        s1 = seg_a; s2 = seg_a + seg_b
        az[s1:s2] = np.random.uniform(0.1, 0.35, seg_b)
        ax[s1:s2] = np.random.normal(0, 0.1, seg_b)
        ay[s1:s2] = np.random.normal(0, 0.1, seg_b)
        gx[s1:s2] = np.random.normal(0, 20, seg_b)
        gy[s1:s2] = np.random.normal(0, 20, seg_b)
        s3 = s2; s4 = s2 + seg_c
        az[s3:s4] = np.random.uniform(4.0, 7.5, seg_c)
        ax[s3:s4] = np.random.uniform(-3, 3, seg_c)
        ay[s3:s4] = np.random.uniform(-3, 3, seg_c)
        gx[s3:s4] = np.random.uniform(-400, 400, seg_c)
        gy[s3:s4] = np.random.uniform(-400, 400, seg_c)
        gz[s3:s4] = np.random.uniform(-400, 400, seg_c)
        az[s4:] = 1.0 + np.random.normal(0, 0.03, n_samples - s4)
        ax[s4:] = np.random.normal(0, 0.03, n_samples - s4)
        ay[s4:] = np.random.normal(0, 0.03, n_samples - s4)
        ax += np.random.normal(0, 0.05, n_samples)
        ay += np.random.normal(0, 0.05, n_samples)
        az += np.random.normal(0, 0.05, n_samples)
        hr   = np.random.uniform(100, 140, n_samples) + np.random.normal(0, 4, n_samples)
        spo2 = np.random.uniform(93, 97, n_samples)   + np.random.normal(0, 0.5, n_samples)
        hrv  = np.random.uniform(10, 30, n_samples)   + np.random.normal(0, 3, n_samples)
        # KEY BMP180 signature: altitude drops ~1 m during a real fall.
        # Pressure rises correspondingly (~0.12 hPa per meter dropped).
        drop_amt = np.random.uniform(0.6, 1.3)
        alt   = np.concatenate([
            np.full(s1, BASE_ALT_M),
            np.linspace(BASE_ALT_M, BASE_ALT_M - drop_amt, s2 - s1),
            np.full(n_samples - s2, BASE_ALT_M - drop_amt),
        ]) + np.random.normal(0, 0.05, n_samples)
        press = BASE_PRESSURE_HPA + (BASE_ALT_M - alt) * 0.12 + np.random.normal(0, 0.15, n_samples)
        # Impact shake propagates to APDS proximity reading too
        prox = np.clip(prox + np.random.randint(0, 30, n_samples), 0, 255)

    elif label == "syncope":
        ax = 0.05 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.04, n_samples)
        ay = 0.05 * np.cos(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.04, n_samples)
        az = 1.0 - 0.4 * (t / t[-1] if t[-1] > 0 else 1) + np.random.normal(0, 0.06, n_samples)
        gx = 20 * np.sin(2 * np.pi * 0.4 * t) + np.random.normal(0, 3, n_samples)
        gy = 15 * np.cos(2 * np.pi * 0.4 * t) + np.random.normal(0, 3, n_samples)
        gz = np.random.normal(0, 3, n_samples)
        hr_start = np.random.uniform(85, 110)
        hr_end   = np.random.uniform(35, 55)
        hr   = np.linspace(hr_start, hr_end, n_samples) + np.random.normal(0, 2, n_samples)
        spo2 = np.random.uniform(88, 95, n_samples) + np.random.normal(0, 0.6, n_samples)
        hrv  = np.random.uniform(8, 25, n_samples)  + np.random.normal(0, 2, n_samples)
        # Gradual altitude drop (postural collapse ~0.4-0.7 m)
        drop_amt = np.random.uniform(0.4, 0.7)
        alt = np.linspace(BASE_ALT_M, BASE_ALT_M - drop_amt, n_samples) + \
              np.random.normal(0, 0.03, n_samples)
        press = BASE_PRESSURE_HPA + (BASE_ALT_M - alt) * 0.12 + np.random.normal(0, 0.1, n_samples)

    seg["ax"] = ax; seg["ay"] = ay; seg["az"] = az
    seg["gx"] = gx; seg["gy"] = gy; seg["gz"] = gz
    seg["hr"]   = np.clip(hr, 0, 220)
    seg["spo2"] = np.clip(spo2, 0, 100)
    seg["hrv"]  = np.clip(hrv, 0, 200)
    seg["press"] = np.clip(press, 900, 1100)
    seg["temp"]  = np.clip(temp,  0, 50)
    seg["alt"]   = alt
    seg["prox"]    = np.clip(prox, 0, 255).astype(int)
    seg["ambient"] = np.clip(ambient, 0, 4000).astype(int)
    seg["red"]     = np.clip(red,   0, 4000).astype(int)
    seg["green"]   = np.clip(green, 0, 4000).astype(int)
    seg["blue"]    = np.clip(blue,  0, 4000).astype(int)
    seg["label"]   = [label] * n_samples
    return seg


def build_dataset(total_minutes=180, out_path="dataset_synthetic.csv"):
    target_samples = total_minutes * 60 * FS
    rows = []
    n_so_far = 0
    classes = list(CLASSES.keys())
    weights = np.array([CLASSES[c]["weight"] for c in classes])
    weights /= weights.sum()

    while n_so_far < target_samples:
        label = np.random.choice(classes, p=weights)
        d_min, d_max = CLASSES[label]["duration_s"]
        dur = np.random.uniform(d_min, d_max)
        n = int(dur * FS)
        if n < 5:
            continue
        seg = gen_segment(label, n)
        for i in range(n):
            rows.append({
                "ax": seg["ax"][i], "ay": seg["ay"][i], "az": seg["az"][i],
                "gx": seg["gx"][i], "gy": seg["gy"][i], "gz": seg["gz"][i],
                "hr":   seg["hr"][i],
                "spo2": seg["spo2"][i],
                "hrv":  seg["hrv"][i],
                "press": seg["press"][i],
                "temp":  seg["temp"][i],
                "alt":   seg["alt"][i],
                "prox":    seg["prox"][i],
                "ambient": seg["ambient"][i],
                "red":     seg["red"][i],
                "green":   seg["green"][i],
                "blue":    seg["blue"][i],
                "label": label,
            })
        n_so_far += n

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} samples ({len(df)/FS/60:.1f} min) -> {out_path}")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    build_dataset(total_minutes=180, out_path="dataset_synthetic.csv")
