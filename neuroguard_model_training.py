# ============================================================
# NeuroGuard v1
# Intelligent Wearable Safety Vest for Neurological Emergencies
# PRODUCTION-GRADE TEMPORAL DEEP LEARNING PIPELINE
# Team SenseSphere - IEEE YESIST12 2026
# ============================================================

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 1. CONFIG
# ============================================================
CONFIG = {
    "csv_path":        "dataset_synthetic.csv",
    "sequence_length": 20,       # 1 second @ 20 Hz (matches firmware)
    "stride":          1,        # sliding window step
    "batch_size":      32,
    "epochs":          100,
    "patience":        12,
    "val_split":       0.15,     # last 15% of time -> val
    "test_split":      0.15,     # last 15% of time -> test
    "out_dir":         "artifacts",
    "model_name":      "neuroguard",
}
os.makedirs(CONFIG["out_dir"], exist_ok=True)

# Raw sensor channels present in the CSV
FEATURES = ["ax", "ay", "az", "gx", "gy", "gz", "hr", "spo2"]
# Environmental channels from BMP180
ENV_FEATURES = ["press", "temp", "alt"]
# Proximity + ambient light from APDS9960
LIGHT_FEATURES = ["prox", "ambient", "red", "green", "blue"]

# ============================================================
# 2. LOAD + FEATURE ENGINEERING
# ============================================================
df = pd.read_csv(CONFIG["csv_path"])
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

# Derived motion features (identical to firmware-side calculation)
if "gyroMag" not in df.columns:
    df["gyroMag"]  = df[["gx", "gy", "gz"]].abs().sum(axis=1)
if "accelMag" not in df.columns:
    df["accelMag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
if "jerk" not in df.columns:
    df["jerk"] = df["accelMag"].diff().abs().fillna(0)
if "hrv" not in df.columns:
    df["hrv"] = df["hr"].rolling(20, min_periods=1).std().fillna(0)
if "roll" not in df.columns:
    df["roll"]  = np.degrees(np.arctan2(df["ay"], df["az"]))
if "pitch" not in df.columns:
    df["pitch"] = np.degrees(np.arctan2(-df["ax"],
                              np.sqrt(df["ay"]**2 + df["az"]**2)))

# Delta features from BMP180 baseline (mean of first 100 rows = calibration proxy)
BASE_PRESS = df["press"].iloc[:100].mean()
BASE_ALT   = df["alt"].iloc[:100].mean()
BASE_TEMP  = df["temp"].iloc[:100].mean()
df["press_delta"] = df["press"] - BASE_PRESS
df["alt_delta"]   = df["alt"]   - BASE_ALT
df["temp_delta"]  = df["temp"]  - BASE_TEMP

# Barometric altitude drop rate (m/s) - falls have strong negative slope
df["alt_slope"] = df["alt_delta"].diff().fillna(0) * 20   # * FS to get per-second

# APDS9960-derived features
# "nocturnal" is a soft binary derived from ambient light; useful for
# disambiguating sleepwalking from normal daytime slow walking.
df["nocturnal"] = (df["ambient"] < 40).astype(float)
# Proximity buildup rate (positive = approaching an obstacle)
df["prox_slope"] = df["prox"].diff().fillna(0) * 20
# Color balance ratios (constant during dark bedroom, variable in daylight)
df["rgb_luma"]  = 0.299 * df["red"] + 0.587 * df["green"] + 0.114 * df["blue"]

# ---- 18-feature set fed to the model ----
USED_FEATURES = [
    # Motion (14)
    "ax", "ay", "az", "gx", "gy", "gz",
    "gyroMag", "accelMag", "jerk", "roll", "pitch",
    "hr", "spo2", "hrv",
    # Environmental BMP180 (2 delta features)
    "press_delta", "alt_delta",
    # APDS9960 (2 features)
    "prox", "ambient",
]
print(f"Using {len(USED_FEATURES)} features: {USED_FEATURES}")

# ============================================================
# 3. LABEL ENCODING
# ============================================================
encoder = LabelEncoder()
df["label_enc"] = encoder.fit_transform(df["label"])
NUM_CLASSES = len(encoder.classes_)
print(f"Classes ({NUM_CLASSES}): {list(encoder.classes_)}")
print("Class distribution:")
print(df["label"].value_counts())

with open(f"{CONFIG['out_dir']}/label_map.json", "w") as f:
    json.dump({int(i): c for i, c in enumerate(encoder.classes_)}, f, indent=2)

# ============================================================
# 4. TEMPORAL SPLIT (NO RANDOM SHUFFLE!)
# ============================================================
n = len(df)
test_idx  = int(n * (1 - CONFIG["test_split"]))
val_idx   = int(n * (1 - CONFIG["test_split"] - CONFIG["val_split"]))

df_train = df.iloc[:val_idx].reset_index(drop=True)
df_val   = df.iloc[val_idx:test_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx:].reset_index(drop=True)
print(f"Split sizes -> train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}")

# ============================================================
# 5. SCALING (fit on TRAIN only - no leakage)
# ============================================================
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(df_train[USED_FEATURES].values)
X_val_flat   = scaler.transform(df_val[USED_FEATURES].values)
X_test_flat  = scaler.transform(df_test[USED_FEATURES].values)

# Save scaler params for embedded inference (firmware_v2_ml_extension.cpp reads these)
np.save(f"{CONFIG['out_dir']}/scaler_mean.npy",  scaler.mean_)
np.save(f"{CONFIG['out_dir']}/scaler_scale.npy", scaler.scale_)

# Also dump as JSON for humans and for the mobile app
with open(f"{CONFIG['out_dir']}/scaler.json", "w") as f:
    json.dump({
        "features": USED_FEATURES,
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }, f, indent=2)

# ============================================================
# 6. SLIDING WINDOW (per split - no temporal leakage)
# ============================================================
def make_windows(X, y, seq_len, stride=1):
    Xs, ys = [], []
    for i in range(0, len(X) - seq_len, stride):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

SEQ = CONFIG["sequence_length"]
STR = CONFIG["stride"]

X_train, y_train_enc = make_windows(X_train_flat, df_train["label_enc"].values, SEQ, STR)
X_val,   y_val_enc   = make_windows(X_val_flat,   df_val["label_enc"].values,   SEQ, STR)
X_test,  y_test_enc  = make_windows(X_test_flat,  df_test["label_enc"].values,  SEQ, STR)

y_train = to_categorical(y_train_enc, NUM_CLASSES)
y_val   = to_categorical(y_val_enc,   NUM_CLASSES)
y_test  = to_categorical(y_test_enc,  NUM_CLASSES)

print(f"Windows -> train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")

# ============================================================
# 7. CLASS WEIGHTS
# ============================================================
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc,
)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}
print(f"Class weights: {class_weights}")

# ============================================================
# 8. MODEL - CNN + BiLSTM + Attention
# ============================================================
def build_model(seq_len, n_features, n_classes):
    inp = Input(shape=(seq_len, n_features), name="window")

    # ---- Multi-scale CNN feature extractor ----
    b1 = layers.Conv1D(32, 3, padding="same", activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    b1 = layers.BatchNormalization()(b1)

    b2 = layers.Conv1D(32, 5, padding="same", activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    b2 = layers.BatchNormalization()(b2)

    x = layers.Concatenate()([b1, b2])
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)

    # ---- Temporal modeling with BiLSTM ----
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2,
                    recurrent_dropout=0.0)
    )(x)

    # ---- Attention over time-steps ----
    attn_scores  = layers.Dense(1, activation="tanh")(x)
    attn_weights = layers.Softmax(axis=1)(attn_scores)
    x = layers.Multiply()([x, attn_weights])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    # ---- Classifier ----
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="state")(x)

    return Model(inp, out, name="NeuroGuardNet")

model = build_model(SEQ, X_train.shape[2], NUM_CLASSES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)
model.summary()

# ============================================================
# 9. CALLBACKS
# ============================================================
ckpt_path = f"{CONFIG['out_dir']}/{CONFIG['model_name']}_best.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["patience"],
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=ckpt_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ============================================================
# 10. TRAINING
# ============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG["epochs"],
    batch_size=CONFIG["batch_size"],
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)

# ============================================================
# 11. EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)
test_metrics = model.evaluate(X_test, y_test, verbose=0)
for name, val in zip(model.metrics_names, test_metrics):
    print(f"{name:>12}: {val:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nPer-class metrics:")
print(classification_report(
    y_true, y_pred,
    target_names=encoder.classes_,
    digits=4,
    zero_division=0,
))

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
print(f"Macro-F1 (best for imbalanced neurological event data): {macro_f1:.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("NeuroGuard - Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{CONFIG['out_dir']}/confusion_matrix.png", dpi=140)
print(f"Saved confusion matrix -> {CONFIG['out_dir']}/confusion_matrix.png")

# Training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="val")
ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{CONFIG['out_dir']}/training_curves.png", dpi=140)

# ============================================================
# 12. SAVE FINAL MODEL (Keras v3 native format)
# ============================================================
final_keras = f"{CONFIG['out_dir']}/{CONFIG['model_name']}.keras"
model.save(final_keras)
print(f"\nSaved final model -> {final_keras}")

# ============================================================
# 13. TFLITE CONVERSION (for ESP32 / mobile deployment)
# ============================================================
print("\nConverting to TFLite (float16) ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
with open(f"{CONFIG['out_dir']}/{CONFIG['model_name']}_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)
print(f"FP16 model size: {len(tflite_fp16)/1024:.1f} KB")

def repr_dataset():
    for i in range(min(200, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

print("Converting to TFLite (int8) ...")
conv8 = tf.lite.TFLiteConverter.from_keras_model(model)
conv8.optimizations = [tf.lite.Optimize.DEFAULT]
conv8.representative_dataset = repr_dataset
conv8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv8.inference_input_type = tf.float32
conv8.inference_output_type = tf.float32
try:
    tflite_int8 = conv8.convert()
    with open(f"{CONFIG['out_dir']}/{CONFIG['model_name']}_int8.tflite", "wb") as f:
        f.write(tflite_int8)
    print(f"INT8 model size: {len(tflite_int8)/1024:.1f} KB")
except Exception as e:
    print(f"INT8 conversion skipped: {e}")

# ============================================================
# 14. REAL-TIME INFERENCE HELPER (with prediction smoothing)
# ============================================================
class NeuroGuardInference:
    """
    Real-time predictor with confidence threshold and
    majority-vote smoothing - prevents flickering predictions
    when the wearer transitions between neurological states.
    """
    def __init__(self, model, scaler_mean, scaler_scale, classes,
                 seq_len=20, smoothing=5, conf_threshold=0.75):
        self.model = model
        self.mean = scaler_mean
        self.scale = scaler_scale
        self.classes = classes
        self.seq_len = seq_len
        self.buf = []
        self.recent = []
        self.smoothing = smoothing
        self.conf_threshold = conf_threshold

    def push(self, feature_row):
        z = (np.asarray(feature_row, dtype=np.float32) - self.mean) / self.scale
        self.buf.append(z)
        if len(self.buf) > self.seq_len:
            self.buf.pop(0)
        if len(self.buf) < self.seq_len:
            return None, 0.0

        x = np.array(self.buf, dtype=np.float32)[None, ...]
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf < self.conf_threshold:
            return "uncertain", conf

        self.recent.append(idx)
        if len(self.recent) > self.smoothing:
            self.recent.pop(0)
        smoothed = max(set(self.recent), key=self.recent.count)
        return self.classes[smoothed], conf


print("\nReal-time inference demo on test set:")
infer = NeuroGuardInference(
    model=model,
    scaler_mean=scaler.mean_,
    scaler_scale=scaler.scale_,
    classes=encoder.classes_,
    seq_len=SEQ,
    smoothing=5,
    conf_threshold=0.75,
)

raw_test = df_test[USED_FEATURES].values
for i in range(min(60, len(raw_test))):
    label, conf = infer.push(raw_test[i])
    if label is not None and i % 10 == 0:
        print(f"  t={i:3d}  pred={label:<22} conf={conf:.2f}")

print("\nDone. All artifacts in:", CONFIG["out_dir"])

# ============================================================
# 15. Subject-aware LOSO validation
# ============================================================
from sklearn.model_selection import LeaveOneGroupOut

def loso_validate(df, USED_FEATURES, build_model_fn, SEQ=20, NUM_CLASSES=8):
    """
    True generalization test: train on N-1 wearers, test on the held-out wearer.
    Report mean +/- std macro-F1.
    """
    if "subject_id" not in df.columns:
        print("No subject_id column - skipping LOSO. Add subject IDs to your CSV.")
        return

    encoder = LabelEncoder()
    df["label_enc"] = encoder.fit_transform(df["label"])

    logo = LeaveOneGroupOut()
    X_all = df[USED_FEATURES].values
    y_all = df["label_enc"].values
    groups = df["subject_id"].values

    f1_scores = []
    for fold, (tr_idx, te_idx) in enumerate(logo.split(X_all, y_all, groups)):
        held_out = df.iloc[te_idx]["subject_id"].iloc[0]
        print(f"\n--- Fold {fold+1}: holding out subject {held_out} ---")

        scaler = StandardScaler().fit(X_all[tr_idx])
        Xtr = scaler.transform(X_all[tr_idx])
        Xte = scaler.transform(X_all[te_idx])

        Xtr_w, ytr_w = make_windows(Xtr, y_all[tr_idx], SEQ)
        Xte_w, yte_w = make_windows(Xte, y_all[te_idx], SEQ)

        model = build_model_fn(SEQ, Xtr_w.shape[2], NUM_CLASSES)
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(Xtr_w, to_categorical(ytr_w, NUM_CLASSES),
                  epochs=30, batch_size=32, verbose=0)

        y_pred = np.argmax(model.predict(Xte_w, verbose=0), axis=1)
        f1 = f1_score(yte_w, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)
        print(f"   Macro-F1 on subject {held_out}: {f1:.4f}")

    print(f"\nLOSO Macro-F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    return f1_scores
