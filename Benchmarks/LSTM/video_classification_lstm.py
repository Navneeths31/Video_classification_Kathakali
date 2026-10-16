# =========================================================
# HEADER FILES
# =========================================================

import os
import warnings
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import (
    Input, Dense, LSTM, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

VIDEO_ROOT      = "/home/navaneeths/Videos_Cropped"

RESULT_LOG_PATH = "/home/navaneeths/Video_Classification/LSTM/results_densenet_lstm.txt"
MODEL_SAVE_PATH = "/home/navaneeths/Video_Classification/LSTM/best_densenet_lstm.keras"

IMG_SIZE     = 224
NUM_FRAMES   = 60
BATCH_SIZE   = 32
EPOCHS       = 50

# =========================================================
# UTILITIES
# =========================================================

def extract_label(video_path):
    """Word label = parent folder name (e.g. Aakasham)."""
    return os.path.basename(os.path.dirname(video_path))


def extract_video_number(video_path):
    """Video number = trailing integer in filename (e.g. Aakasham_7 → 7)."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        print(f"[WARN] Cannot parse video number from: {video_path}")
        return -1


def sample_frames(video_path, num_frames=NUM_FRAMES):
    """
    Uniformly sample `num_frames` frames from a video.
    Returns float32 array of shape (num_frames, IMG_SIZE, IMG_SIZE, 3),
    pre-processed for VGG19 (values in [-1, 1]).
    If the video has fewer frames than requested, the last frame is repeated.
    """
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        print(f"[WARN] Could not read frames from: {video_path}")
        return np.zeros((num_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            # Repeat last valid frame if seek fails
            frames.append(frames[-1] if frames else
                          np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = preprocess_input(frame.astype(np.float32))  # [-1, 1]
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)   # (num_frames, 224, 224, 3)

# =========================================================
# DISCOVER VIDEOS
# =========================================================

print("\nScanning video directory...")

video_paths = []
for root, dirs, files in os.walk(VIDEO_ROOT):
    for fname in files:
        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_paths.append(os.path.join(root, fname))

video_paths.sort()

print(f"Found {len(video_paths)} videos.")

labels       = [extract_label(p)        for p in video_paths]
video_numbers = [extract_video_number(p) for p in video_paths]

# Filter out any paths where number parsing failed
valid = [(p, l, n) for p, l, n in zip(video_paths, labels, video_numbers)
         if n != -1]

video_paths, labels, video_numbers = zip(*valid) if valid else ([], [], [])
video_paths    = list(video_paths)
labels         = list(labels)
video_numbers  = list(video_numbers)

print(f"Valid videos after filtering: {len(video_paths)}")
print(f"Classes found: {sorted(set(labels))}")

# =========================================================
# LABEL ENCODING
# =========================================================

le          = LabelEncoder()
y           = le.fit_transform(labels)
num_classes = len(le.classes_)
y_cat       = to_categorical(y, num_classes)

print(f"\nNumber of classes: {num_classes}")

# =========================================================
# TRAIN / VAL / TEST SPLIT  (same convention as other scripts)
#   Videos 1-7  → train
#   Video  8    → val
#   Videos 9-10 → test
# =========================================================

video_numbers = np.array(video_numbers)

train_mask = np.isin(video_numbers, [1, 2, 3, 4, 5, 6, 7])
val_mask   = np.isin(video_numbers, [8])
test_mask  = np.isin(video_numbers, [9, 10])

def apply_mask(lst, mask):
    return [v for v, m in zip(lst, mask) if m]

train_paths  = apply_mask(video_paths, train_mask)
val_paths    = apply_mask(video_paths, val_mask)
test_paths   = apply_mask(video_paths, test_mask)

y_train = y_cat[train_mask]
y_val   = y_cat[val_mask]
y_test  = y_cat[test_mask]

y_train_labels = y[train_mask]
y_val_labels   = y[val_mask]
y_test_labels  = y[test_mask]

print(f"\nTrain videos : {len(train_paths)}")
print(f"Val videos   : {len(val_paths)}")
print(f"Test videos  : {len(test_paths)}")

# =========================================================
# RESNET50 FEATURE EXTRACTOR  (frozen)
# =========================================================

print("\nBuilding VGG19 feature extractor...")

vgg_base = DenseNet121(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"            # GlobalAveragePooling2D built-in → 1280-d
)
vgg_base.trainable = False

# =========================================================
# FRAME FEATURE EXTRACTION
# =========================================================

def extract_features_for_paths(paths, desc=""):
    """
    For each video, sample NUM_FRAMES frames and extract
    VGG19 embeddings. Returns array of shape
    (num_videos, NUM_FRAMES, 1280).
    """
    all_features = []
    total = len(paths)

    for i, path in enumerate(paths):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{desc}] {i+1}/{total} — {os.path.basename(path)}")

        frames = sample_frames(path)             # (30, 224, 224, 3)
        feats  = vgg_base.predict(
            frames, batch_size=NUM_FRAMES, verbose=0
        )                                        # (30, 1024)
        all_features.append(feats)

    return np.array(all_features, dtype=np.float32)


print("\nExtracting train features...")
X_train = extract_features_for_paths(train_paths, desc="Train")

print("\nExtracting val features...")
X_val   = extract_features_for_paths(val_paths,   desc="Val")

print("\nExtracting test features...")
X_test  = extract_features_for_paths(test_paths,  desc="Test")

print(f"\nX_train shape : {X_train.shape}")   # (N_train, 30, 1280)
print(f"X_val shape   : {X_val.shape}")
print(f"X_test shape  : {X_test.shape}")

# =========================================================
# MODEL  —  Single LSTM → Dense classifier
# =========================================================
#
#   Input  (30, 1280)
#     │
#   LSTM  512 units  (return_sequences=False)
#     │
#   BatchNormalization
#     │
#   Dense  256 → BN → Dense num_classes → Softmax
#
# =========================================================

sequence_input = Input(shape=(NUM_FRAMES, 1024), name="sequence_input")

x = LSTM(512, return_sequences=False)(sequence_input)
x = BatchNormalization()(x)

x = Dense(256)(x)
x = BatchNormalization()(x)

class_output = Dense(num_classes, activation="softmax",
                     name="classification")(x)

model = Model(inputs=sequence_input, outputs=class_output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── CHANGE ③: Cosine decay LR schedule with linear warmup ─
# Warmup for 5 epochs prevents the Adam momentum from
# blowing through the shallow optimum early in training.
# After warmup, cosine decay smoothly lowers LR to ~0
# by epoch 50, avoiding the sharp loss spikes that come
# from a flat LR hitting a narrow valley.
WARMUP_EPOCHS = 5
INIT_LR       = 1e-4
MIN_LR        = 1e-6
steps_per_epoch = len(train_paths) // BATCH_SIZE
total_steps  = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, min_lr, warmup_steps, total_steps):
        self.init_lr      = init_lr
        self.min_lr       = min_lr
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps  = tf.cast(total_steps,  tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup
        warmup_lr = self.init_lr * (step / self.warmup_steps)
        # Cosine decay
        progress  = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos_decay = 0.5 * (1.0 + tf.math.cos(np.pi * tf.clip_by_value(progress, 0.0, 1.0)))
        cosine_lr = self.min_lr + (self.init_lr - self.min_lr) * cos_decay
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "init_lr":      self.init_lr,
            "min_lr":       self.min_lr,
            "warmup_steps": int(self.warmup_steps.numpy()),
            "total_steps":  int(self.total_steps.numpy()),
        }

lr_schedule = WarmupCosineDecay(INIT_LR, MIN_LR, warmup_steps, total_steps)

# Recompile with the new schedule (loss unchanged from above)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# =========================================================
# CALLBACKS
# =========================================================

checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


class EpochLogger(tf.keras.callbacks.Callback):

    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        msg = (
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Acc: {logs.get('accuracy', 0):.4f} | "
            f"Val Acc: {logs.get('val_accuracy', 0):.4f} | "
            f"Train Loss: {logs.get('loss', 0):.4f} | "
            f"Val Loss: {logs.get('val_loss', 0):.4f}\n"
        )

        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg)


epoch_logger = EpochLogger(RESULT_LOG_PATH)

# =========================================================
# TRAIN
# =========================================================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, epoch_logger]
)

# =========================================================
# PREDICT
# =========================================================

pred_probs = model.predict(X_test)
preds      = np.argmax(pred_probs, axis=1)

# =========================================================
# OUTPUT + LOGS
# =========================================================

with open(RESULT_LOG_PATH, "a", encoding="utf-8") as f:

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("FINAL RESULTS\n")
    f.write("=" * 80 + "\n\n")

    acc     = accuracy_score(y_test_labels, preds)
    acc_msg = (
        "\n" + "=" * 60 + "\n"
        f"TEST ACCURACY: {acc:.4f}\n"
        + "=" * 60 + "\n"
    )

    print(acc_msg)
    f.write(acc_msg + "\n")

    labels_present       = np.unique(np.concatenate([y_test_labels, preds]))
    target_names_present = le.inverse_transform(labels_present)

    report = classification_report(
        y_test_labels, preds,
        labels=labels_present,
        target_names=target_names_present,
        zero_division=0
    )

    print(report)

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 80 + "\n\n")
    f.write(report + "\n\n")

    best_epoch      = np.argmax(history.history["val_accuracy"]) + 1
    best_acc        = np.max(history.history["val_accuracy"])
    best_train_acc  = history.history["accuracy"][best_epoch - 1]
    best_train_loss = history.history["loss"][best_epoch - 1]
    best_val_loss   = history.history["val_loss"][best_epoch - 1]

    best_msg = (
        "\n" + "=" * 80 + "\n"
        "BEST MODEL DETAILS\n"
        + "=" * 80 + "\n\n"
        f"Best Epoch       : {best_epoch}\n"
        f"Best Val Accuracy: {best_acc:.4f}\n"
        f"Train Acc @ Best : {best_train_acc:.4f}\n"
        f"Train Loss @ Best: {best_train_loss:.4f}\n"
        f"Val Loss @ Best  : {best_val_loss:.4f}\n\n"
        f"Saved Model Path : {MODEL_SAVE_PATH}\n"
    )

    print(best_msg)
    f.write(best_msg)

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("FULL TRAINING HISTORY\n")
    f.write("=" * 80 + "\n\n")

    for epoch in range(EPOCHS):
        f.write(
            f"Epoch {epoch+1:02d} | "
            f"Train Acc: {history.history['accuracy'][epoch]:.4f} | "
            f"Val Acc: {history.history['val_accuracy'][epoch]:.4f} | "
            f"Train Loss: {history.history['loss'][epoch]:.4f} | "
            f"Val Loss: {history.history['val_loss'][epoch]:.4f}\n"
        )

print("\nLogs saved to:")
print(RESULT_LOG_PATH)
