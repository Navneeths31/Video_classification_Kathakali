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
    Input, Dense, BatchNormalization, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from transformers import AutoImageProcessor, AutoModel   # ← generic HF API
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

VIDEO_ROOT      = "/home/navaneeths/Videos_Cropped"

RESULT_LOG_PATH = "/home/navaneeths/Video_Classification/Transformers/results_log_swin.txt"
MODEL_SAVE_PATH = "/home/navaneeths/Video_Classification/Transformers/best_swin.keras"

IMG_SIZE     = 224
NUM_FRAMES   = 60
BATCH_SIZE   = 32
EPOCHS       = 50

# ── Swin-Base outputs 1024-d pooled features (ViT-Base was 768) ──
SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
VIT_EMBED_DIM   = 1024     # <-- updated from 768

# Temporal transformer encoder config (unchanged)
NUM_HEADS = 8
FF_DIM    = 512

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
    Uniformly sample `num_frames` RGB frames from a video.
    Returns uint8 array of shape (num_frames, IMG_SIZE, IMG_SIZE, 3).
    If fewer frames than requested, last frame is repeated.
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        print(f"[WARN] Could not read frames from: {video_path}")
        return np.zeros((num_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            frames.append(frames[-1] if frames else
                          np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame.astype(np.uint8))

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

labels        = [extract_label(p)        for p in video_paths]
video_numbers = [extract_video_number(p) for p in video_paths]

valid = [(p, l, n) for p, l, n in zip(video_paths, labels, video_numbers)
         if n != -1]

video_paths, labels, video_numbers = zip(*valid) if valid else ([], [], [])
video_paths   = list(video_paths)
labels        = list(labels)
video_numbers = list(video_numbers)

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
# TRAIN / VAL / TEST SPLIT
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

train_paths = apply_mask(video_paths, train_mask)
val_paths   = apply_mask(video_paths, val_mask)
test_paths  = apply_mask(video_paths, test_mask)

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
# SWIN TRANSFORMER FEATURE EXTRACTOR  (frozen, HuggingFace)
# =========================================================

print(f"\nLoading pretrained Swin Transformer ({SWIN_MODEL_NAME})...")

# AutoFeatureExtractor handles Swin's image pre-processing automatically
feature_extractor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)
swin_model        = AutoModel.from_pretrained(SWIN_MODEL_NAME)
swin_model.eval()   # frozen — no gradient updates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model.to(device)

print(f"Swin Transformer running on: {device}")

# =========================================================
# FRAME FEATURE EXTRACTION
# =========================================================

def extract_swin_features(paths, desc=""):
    """
    For each video:
      1. Sample NUM_FRAMES frames  → (NUM_FRAMES, 224, 224, 3)
      2. Pass each frame through frozen Swin Transformer
      3. Take pooler_output (global avg of patch tokens) → 1024-d per frame
         Note: Swin has NO [CLS] token, so we use pooler_output instead.
    Returns array of shape (num_videos, NUM_FRAMES, VIT_EMBED_DIM).
    """
    all_features = []
    total        = len(paths)

    with torch.no_grad():
        for i, path in enumerate(paths):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{desc}] {i+1}/{total} — {os.path.basename(path)}")

            frames = sample_frames(path)   # (NUM_FRAMES, 224, 224, 3) uint8

            # Process all frames in one batch
            inputs = feature_extractor(
                images=list(frames),
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = swin_model(**inputs)

            # pooler_output: global average of all patch token embeddings
            # shape → (NUM_FRAMES, 1024)
            frame_embeddings = outputs.pooler_output   # (NUM_FRAMES, 1024)
            frame_embeddings = frame_embeddings.cpu().numpy().astype(np.float32)

            all_features.append(frame_embeddings)

    return np.array(all_features, dtype=np.float32)


print("\nExtracting train features...")
X_train = extract_swin_features(train_paths, desc="Train")

print("\nExtracting val features...")
X_val   = extract_swin_features(val_paths,   desc="Val")

print("\nExtracting test features...")
X_test  = extract_swin_features(test_paths,  desc="Test")

print(f"\nX_train shape : {X_train.shape}")   # (N_train, NUM_FRAMES, 1024)
print(f"X_val shape   : {X_val.shape}")
print(f"X_test shape  : {X_test.shape}")

# =========================================================
# TEMPORAL TRANSFORMER ENCODER BLOCK  (unchanged)
# =========================================================
#
#   Input  (NUM_FRAMES, 1024)
#     │
#   MultiHeadAttention (4 heads)  +  residual  →  LayerNorm
#     │
#   Dense FF (512 → 1024)         +  residual  →  LayerNorm
#     │
#   GlobalAveragePooling1D  →  1024-d
#     │
#   Dense 256 → BN
#     │
#   Dense num_classes → Softmax
#
# =========================================================

sequence_input = Input(shape=(NUM_FRAMES, VIT_EMBED_DIM),
                        name="sequence_input")

# --- Multi-Head Self-Attention + residual + LayerNorm ---
attn_output = MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=VIT_EMBED_DIM // NUM_HEADS   # 256 (1024 / 4)
)(sequence_input, sequence_input)

x = Add()([sequence_input, attn_output])
x = LayerNormalization(epsilon=1e-6)(x)

# --- Feed-Forward + residual + LayerNorm ---
ff = Dense(FF_DIM, activation="relu")(x)
ff = Dense(VIT_EMBED_DIM)(ff)            # project back to 1024

x = Add()([x, ff])
x = LayerNormalization(epsilon=1e-6)(x)

# --- Aggregate sequence → single vector ---
x = GlobalAveragePooling1D()(x)

# --- Classifier head ---
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
