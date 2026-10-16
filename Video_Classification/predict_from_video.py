# =========================================================
# PREDICT FROM RAW VIDEO
# =========================================================

import os
import sys
sys.path.insert(0, r"C:\Users\navne\OneDrive\Desktop\DWPose-onnx\ControlNet-v1-1-nightly")
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

KEYFRAME_MODEL_PATH   = r"C:\Users\navne\Pictures\Kathakali\Best_Models\best_MobileNetV2_v9.pth"
CLASSIFIER_MODEL_PATH = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Our_model\best_fusion.keras"
CSV_PATH              = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_keypoints.csv"

IMG_SIZE     = 224
BATCH_SIZE   = 32
TOTAL_KPS    = 50
TARGET_FPS   = 5
KF_THRESHOLD = 0.5

TARGET_W = 2300
TARGET_H = 2160

# =========================================================
# KEYPOINT LAYOUT
# =========================================================

KEEP_INDICES = [0, 1, 2, 3, 4, 5, 6, 7,
                *range(92, 113),
                *range(113, 134)]

KEYPOINT_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    *[f"l_hand_{i}" for i in range(21)],
    *[f"r_hand_{i}" for i in range(21)],
]

KP_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# =========================================================
# LIMB CHAIN
# =========================================================

LIMB_CHAIN = [
    ("neck_r_shoulder",  "neck",        "r_shoulder"),
    ("neck_l_shoulder",  "neck",        "l_shoulder"),
    ("r_upper_arm",      "r_shoulder",  "r_elbow"),
    ("r_forearm",        "r_elbow",     "r_wrist"),
    ("r_wrist_hand",     "r_wrist",     "r_hand_0"),
    ("l_upper_arm",      "l_shoulder",  "l_elbow"),
    ("l_forearm",        "l_elbow",     "l_wrist"),
    ("l_wrist_hand",     "l_wrist",     "l_hand_0"),
    ("l_thumb_1",  "l_hand_0", "l_hand_1"),
    ("l_thumb_2",  "l_hand_1", "l_hand_2"),
    ("l_thumb_3",  "l_hand_2", "l_hand_3"),
    ("l_thumb_4",  "l_hand_3", "l_hand_4"),
    ("l_index_1",  "l_hand_0", "l_hand_5"),
    ("l_index_2",  "l_hand_5", "l_hand_6"),
    ("l_index_3",  "l_hand_6", "l_hand_7"),
    ("l_index_4",  "l_hand_7", "l_hand_8"),
    ("l_middle_1", "l_hand_0", "l_hand_9"),
    ("l_middle_2", "l_hand_9", "l_hand_10"),
    ("l_middle_3", "l_hand_10","l_hand_11"),
    ("l_middle_4", "l_hand_11","l_hand_12"),
    ("l_ring_1",   "l_hand_0", "l_hand_13"),
    ("l_ring_2",   "l_hand_13","l_hand_14"),
    ("l_ring_3",   "l_hand_14","l_hand_15"),
    ("l_ring_4",   "l_hand_15","l_hand_16"),
    ("l_pinky_1",  "l_hand_0", "l_hand_17"),
    ("l_pinky_2",  "l_hand_17","l_hand_18"),
    ("l_pinky_3",  "l_hand_18","l_hand_19"),
    ("l_pinky_4",  "l_hand_19","l_hand_20"),
    ("r_thumb_1",  "r_hand_0", "r_hand_1"),
    ("r_thumb_2",  "r_hand_1", "r_hand_2"),
    ("r_thumb_3",  "r_hand_2", "r_hand_3"),
    ("r_thumb_4",  "r_hand_3", "r_hand_4"),
    ("r_index_1",  "r_hand_0", "r_hand_5"),
    ("r_index_2",  "r_hand_5", "r_hand_6"),
    ("r_index_3",  "r_hand_6", "r_hand_7"),
    ("r_index_4",  "r_hand_7", "r_hand_8"),
    ("r_middle_1", "r_hand_0", "r_hand_9"),
    ("r_middle_2", "r_hand_9", "r_hand_10"),
    ("r_middle_3", "r_hand_10","r_hand_11"),
    ("r_middle_4", "r_hand_11","r_hand_12"),
    ("r_ring_1",   "r_hand_0", "r_hand_13"),
    ("r_ring_2",   "r_hand_13","r_hand_14"),
    ("r_ring_3",   "r_hand_14","r_hand_15"),
    ("r_ring_4",   "r_hand_15","r_hand_16"),
    ("r_pinky_1",  "r_hand_0", "r_hand_17"),
    ("r_pinky_2",  "r_hand_17","r_hand_18"),
    ("r_pinky_3",  "r_hand_18","r_hand_19"),
    ("r_pinky_4",  "r_hand_19","r_hand_20"),
]

# =========================================================
# ANGLE DEFINITIONS
# =========================================================

ANGLE_TRIPLETS = [
    (2, 3, 4), (5, 6, 7),
    (3, 2, 5), (6, 5, 2),
    (4, 2, 5), (7, 5, 2),
    (8, 9, 10),   (9, 10, 11),   (10, 11, 12),
    (8, 13, 14),  (13, 14, 15),  (14, 15, 16),
    (8, 17, 18),  (17, 18, 19),  (18, 19, 20),
    (8, 21, 22),  (21, 22, 23),  (22, 23, 24),
    (8, 25, 26),  (25, 26, 27),  (26, 27, 28),
    (29, 30, 31), (30, 31, 32),  (31, 32, 33),
    (29, 34, 35), (34, 35, 36),  (35, 36, 37),
    (29, 38, 39), (38, 39, 40),  (39, 40, 41),
    (29, 42, 43), (42, 43, 44),  (43, 44, 45),
    (29, 46, 47), (46, 47, 48),  (47, 48, 49),
    (9, 8, 13),   (13, 8, 17),  (17, 8, 21),  (21, 8, 25),
    (30, 29, 34), (34, 29, 38), (38, 29, 42), (42, 29, 46),
]

# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def compute_limb(p1, p2, shoulder_width):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx * dx + dy * dy)
    normalized_length = (length / shoulder_width) * 0.1 if shoulder_width > 0 else 0.0
    angle = np.arctan2(dy, dx)
    return normalized_length, angle


def reconstruct_point(parent, norm_length, angle):
    x = parent[0] + norm_length * np.cos(angle)
    y = parent[1] + norm_length * np.sin(angle)
    return np.array([x, y], dtype=np.float32)


def normalize_keypoints(keypoints, scores, W, H):
    """
    Apply the same normalization pipeline as extract_pose_normalized.py:
    - Translate so neck is at (0,0)
    - Scale by image dimensions
    - Normalize limb lengths by shoulder width * 0.1
    - Reconstruct skeleton
    Returns reconstructed_keypoints (50, 2) or raises Exception.
    """
    # Translate so neck is at (0, 0)
    neck_x = keypoints[0, 1, 0]
    neck_y = keypoints[0, 1, 1]

    if neck_x == 0 and neck_y == 0:
        raise Exception("Neck not detected")

    keypoints[..., :2] -= np.array([neck_x, neck_y])
    
    translation = np.array([neck_x, neck_y])
    keypoints[..., :2] -= translation

    # Scale by image dimensions
    keypoints[..., 0] /= float(W)
    keypoints[..., 1] /= float(H)

    flattened = keypoints[0]

    # Shoulder width
    r_shoulder_pt = flattened[KP_IDX["r_shoulder"]]
    l_shoulder_pt = flattened[KP_IDX["l_shoulder"]]
    shoulder_width = np.linalg.norm(r_shoulder_pt - l_shoulder_pt)

    if shoulder_width < 1e-6:
        raise Exception("Shoulder width too small")

    # Compute limb features
    limb_features = {}
    for (feat_name, p1_name, p2_name) in LIMB_CHAIN:
        p1 = flattened[KP_IDX[p1_name]]
        p2 = flattened[KP_IDX[p2_name]]
        norm_len, angle = compute_limb(p1, p2, shoulder_width)
        limb_features[feat_name] = (norm_len, angle)

    nose_pt = flattened[KP_IDX["nose"]]
    neck_pt = flattened[KP_IDX["neck"]]
    nose_norm_len, nose_angle = compute_limb(neck_pt, nose_pt, shoulder_width)

    # Reconstruct from neck = (0, 0)
    reconstructed = {}
    reconstructed["neck"] = np.array([0.0, 0.0], dtype=np.float32)
    reconstructed["nose"] = reconstruct_point(reconstructed["neck"], nose_norm_len, nose_angle)

    for (feat_name, p1_name, p2_name) in LIMB_CHAIN:
        norm_len, angle = limb_features[feat_name]
        reconstructed[p2_name] = reconstruct_point(reconstructed[p1_name], norm_len, angle)

    # Build keypoint array
    reconstructed_keypoints = np.zeros((len(KEYPOINT_NAMES), 2), dtype=np.float32)
    for i, name in enumerate(KEYPOINT_NAMES):
        if name in reconstructed:
            reconstructed_keypoints[i] = reconstructed[name]
        else:
            reconstructed_keypoints[i] = flattened[i]

    return reconstructed_keypoints, flattened


def draw_normalized_skeleton(reconstructed_keypoints, full_scores, H, W):
    """
    Draw skeleton from reconstructed keypoints exactly as in extract_pose_normalized.py.
    """
    from annotator.dwpose import draw_pose

    full_candidate = np.full((1, 134, 2), -1, dtype=np.float32)

    for out_idx, orig_idx in enumerate(KEEP_INDICES):
        full_candidate[0, orig_idx, :] = reconstructed_keypoints[out_idx]

    # Shift only filled slots so neck (0,0) maps to canvas centre (0.5, 0.5)
    for orig_idx in KEEP_INDICES:
        full_candidate[0, orig_idx, :] += 0.5

    body = full_candidate[:, :18].copy()
    body = body.reshape(1 * 18, 2)

    score = full_scores[:, :18].copy()
    for j in range(18):
        if body[j, 0] < 0 or body[j, 1] < 0:
            score[0, j] = -1
        elif score[0, j] > 0.3:
            score[0, j] = j
        else:
            score[0, j] = -1

    hands = full_candidate[:, 92:113]
    hands = np.vstack([hands, full_candidate[:, 113:]])
    faces = full_candidate[:, 24:92]

    bodies = dict(candidate=body, subset=score)
    pose   = dict(bodies=bodies, hands=hands, faces=faces)

    return draw_pose(pose, H, W)


# =========================================================
# STAGE 1 — KEYFRAME EXTRACTION
# =========================================================

def load_keyframe_model(path):
    import torch
    import torch.nn as nn
    from torchvision import models
    from transformers import CLIPVisionModel

    class KeyframeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            for p in self.clip.parameters():
                p.requires_grad = False
            mobilenet     = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.backbone = mobilenet.features
            self.pool     = nn.AdaptiveAvgPool2d((1, 1))
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.fc   = nn.Linear(1280 + 768, 512)
            self.lstm = nn.LSTM(512, 512, batch_first=True, bidirectional=True, num_layers=2)
            self.head = nn.Sequential(
    		nn.Linear(1024, 256),
    		nn.ReLU(),
    		nn.Dropout(0.3),
    		nn.Linear(256, 1)
	    )    

        def forward(self, x):
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            with torch.no_grad():
                cnn  = self.pool(self.backbone(x)).flatten(1)
                clip = self.clip(pixel_values=x).pooler_output
            fused = torch.cat([cnn, clip], dim=1)
            x     = torch.relu(self.fc(fused)).view(b, t, -1)
            x, _  = self.lstm(x)
            return self.head(x).squeeze(-1)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = KeyframeModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[Stage 1] Keyframe model loaded.")
    return model, device


def extract_keyframes(video_path, kf_model, device):
    import torch
    import torchvision.transforms as T

    norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])

    cap      = cv2.VideoCapture(video_path)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / TARGET_FPS))
    indices  = list(range(0, total, interval))

    frames_bgr, frames_t = [], []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames_bgr.append(f.copy())
        else:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            frames_bgr.append(np.zeros((224, 224, 3), dtype=np.uint8))

        resized = cv2.resize(rgb, (224, 224))
        t = norm(torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0)
        frames_t.append(t)

    cap.release()

    if not frames_t:
        print("[Stage 1] No frames could be read!")
        return []

    vid_tensor = torch.stack(frames_t).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = torch.sigmoid(kf_model(vid_tensor)).squeeze(0).cpu().numpy()

    keyframes = [
        (indices[i], frames_bgr[i])
        for i, score in enumerate(scores)
        if score >= KF_THRESHOLD
    ]

    print(f"[Stage 1] Extracted {len(keyframes)} keyframes "
          f"(score >= {KF_THRESHOLD}) from {len(indices)} sampled frames.")
    return keyframes


# =========================================================
# STAGE 2 — DWPose + NORMALIZED SKELETONISATION
# =========================================================

def run_dwpose(keyframes, skeletal_dir, original_frames_dir, json_path):
    from annotator.dwpose.wholebody import Wholebody

    pose_estimator = Wholebody()

    os.makedirs(skeletal_dir, exist_ok=True)
    os.makedirs(original_frames_dir, exist_ok=True)

    skeletal_paths = []
    kp_records     = []

    for frame_idx, bgr in keyframes:

        # Save original frame
        orig_path = os.path.join(original_frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(orig_path, bgr)

        # Pad to 2300x2160
        h, w = bgr.shape[:2]
        pad_top    = (TARGET_H - h) // 2
        pad_bottom = TARGET_H - h - pad_top
        pad_left   = (TARGET_W - w) // 2
        pad_right  = TARGET_W - w - pad_left

        if pad_top < 0 or pad_bottom < 0 or pad_left < 0 or pad_right < 0:
            # Image larger than target — resize down to fit
            bgr = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            pad_top = pad_bottom = pad_left = pad_right = 0

        padded = cv2.copyMakeBorder(
            bgr,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        H, W, _ = padded.shape

        try:
            keypoints, scores = pose_estimator(padded)
            full_scores = scores.copy()

            if len(keypoints) == 0:
                raise Exception("No person detected")

            keypoints = keypoints[:, KEEP_INDICES, :]
            scores    = scores[:, KEEP_INDICES]

            # Normalize keypoints using same pipeline as training
            reconstructed_keypoints, flattened = normalize_keypoints(
                keypoints, scores, W, H
            )

            # Draw normalized skeleton
            skeletal_image = draw_normalized_skeleton(reconstructed_keypoints, full_scores, H, W)

            save_path = os.path.join(skeletal_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(save_path, skeletal_image)
            skeletal_paths.append(save_path)

            # Save keypoint record (reconstructed coords)
            record = {"frame_index": frame_idx, "keypoints": {}}
            for i, name in enumerate(KEYPOINT_NAMES):
                record["keypoints"][name] = {
                    "x": float(reconstructed_keypoints[i][0]),
                    "y": float(reconstructed_keypoints[i][1])
                }
            kp_records.append(record)

        except Exception as e:
            print(f"[Stage 2] Failed on frame {frame_idx}: {e}")
            blank     = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
            save_path = os.path.join(skeletal_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(save_path, blank)
            skeletal_paths.append(save_path)

            record = {
                "frame_index": frame_idx,
                "keypoints": {
                    name: {"x": 0.0, "y": 0.0}
                    for name in KEYPOINT_NAMES
                }
            }
            kp_records.append(record)

    with open(json_path, "w") as f:
        json.dump(kp_records, f, indent=2)

    print(f"[Stage 2] {len(skeletal_paths)} skeletal images saved.")
    print(f"[Stage 2] Keypoints saved to:\n  {json_path}")
    return skeletal_paths, kp_records


# =========================================================
# STAGE 3 — FEATURE ENGINEERING + CLASSIFICATION
# =========================================================

def get_xy(kps_flat, idx):
    return kps_flat[idx * 2], kps_flat[idx * 2 + 1]

def euclidean(x1, y1, x2, y2):
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def angle_at_b(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by], dtype=np.float32)
    bc = np.array([cx - bx, cy - by], dtype=np.float32)
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-6 or n_bc < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def engineer_features(row_values):
    kps   = np.array(row_values, dtype=np.float32)
    feats = list(kps)
    for (a, b, c) in ANGLE_TRIPLETS:
        if max(a, b, c) >= TOTAL_KPS:
            feats.append(0.0)
            continue
        ax, ay = get_xy(kps, a)
        bx, by = get_xy(kps, b)
        cx, cy = get_xy(kps, c)
        feats.append(angle_at_b(ax, ay, bx, by, cx, cy) / 180.0)
    r_root = get_xy(kps, 8)
    l_root = get_xy(kps, 29)
    for tip_idx in [12, 16, 20, 24, 28]:
        tx, ty = get_xy(kps, tip_idx)
        feats.append(euclidean(tx, ty, *r_root))
    for tip_idx in [33, 37, 41, 45, 49]:
        tx, ty = get_xy(kps, tip_idx)
        feats.append(euclidean(tx, ty, *l_root))
    return np.array(feats, dtype=np.float32)


def kp_record_to_flat(record):
    flat = []
    for name in KEYPOINT_NAMES:
        kp = record["keypoints"].get(name, {"x": 0.0, "y": 0.0})
        flat += [kp["x"], kp["y"]]
    return flat


# =========================================================
# WARMUP + COSINE DECAY LR SCHEDULE
# (must match the class used when CLASSIFIER_MODEL_PATH was
#  trained/saved, so Keras can deserialize the model's
#  optimizer config on load)
# =========================================================

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


def load_classifier_model(path):
    """
    Load the fusion classifier, making the custom WarmupCosineDecay
    schedule available for deserialization. compile=False is used
    since only inference (forward pass) is needed here — no
    optimizer/training state has to be reconstructed.
    """
    print(f"[Stage 3] Loading classifier from:\n  {path}")
    model = tf.keras.models.load_model(
        path,
        custom_objects={"WarmupCosineDecay": WarmupCosineDecay},
        compile=False
    )
    return model


def load_scaler_and_le():
    print("[Stage 3] Fitting scaler from training CSV...")
    df = pd.read_csv(CSV_PATH)

    def _vname(p): return os.path.basename(os.path.dirname(str(p).replace("\\", "/")))
    def _wname(p): return os.path.basename(os.path.dirname(os.path.dirname(str(p).replace("\\", "/"))))
    def _vnum(v):
        try: return int(str(v).split("_")[-1])
        except: return -1

    df["video_name"]   = df["skeletal_image_path"].apply(_vname)
    df["word_name"]    = df["skeletal_image_path"].apply(_wname)
    df["video_number"] = df["video_name"].apply(_vnum)
    df = df[df["video_number"] != -1].copy()

    drop_cols  = ["image_path", "skeletal_image_path",
                  "video_name", "video_number", "word_name", "label"]
    coord_cols = [c for c in df.columns if c not in drop_cols]

    X_all = np.vstack([
        engineer_features(row)
        for row in df[coord_cols].replace(-1, 0).values
    ])

    train_mask = np.isin(df["video_number"].values, [1,2,3,4,5,6,7,8])
    scaler     = StandardScaler()
    scaler.fit(X_all[train_mask])
    X_scaled   = scaler.transform(X_all)

    df_corr  = pd.DataFrame(X_scaled[train_mask])
    corr_mat = df_corr.corr().abs()
    upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop  = [col for col in upper.columns if any(upper[col] > 0.98)]

    le = LabelEncoder()
    le.fit(df["label"].values)

    print(f"[Stage 3] Classes: {len(le.classes_)}  |  "
          f"Features after drop: {X_all.shape[1] - len(to_drop)}")
    return scaler, to_drop, le


def classify_frames(keras_model, scaler, to_drop, le, skeletal_paths, kp_records):

    X_pose = np.vstack([
        engineer_features(kp_record_to_flat(r))
        for r in kp_records
    ])
    X_pose = scaler.transform(X_pose)
    X_pose = pd.DataFrame(X_pose).drop(columns=to_drop).values.astype(np.float32)

    def _load_image(path):
        # Must match training's parse_function exactly: the CNN branch
        # is a custom from-scratch CNN (not a pretrained MobileNetV2
        # backbone), so it expects images rescaled to [0, 1], NOT
        # ImageNet-style mobilenet_v2.preprocess_input (~[-1, 1]).
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        return img

    all_probs = []
    for start in range(0, len(skeletal_paths), BATCH_SIZE):
        batch_paths = skeletal_paths[start:start + BATCH_SIZE]
        batch_pose  = X_pose[start:start + BATCH_SIZE]
        imgs  = tf.stack([_load_image(p) for p in batch_paths]).numpy()
        probs = keras_model.predict(
            {"image_input": imgs, "pose_input": batch_pose},
            verbose=0
        )
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    TOP_N = 1
    top_n_indices = np.argsort(all_probs, axis=1)[:, -TOP_N:][:, ::-1]
    pred_names    = le.inverse_transform(top_n_indices[:, 0]).tolist()

    top_n_predictions = []
    for i in range(len(all_probs)):
        frame_preds = []
        for rank in range(TOP_N):
            cls_idx = top_n_indices[i, rank]
            frame_preds.append({
                "class": le.inverse_transform([cls_idx])[0],
                "prob":  float(all_probs[i, cls_idx])
            })
        top_n_predictions.append(frame_preds)

    print(f"[Stage 3] Classified {len(pred_names)} frames.")
    return pred_names, top_n_predictions


# =========================================================
# STAGE 4 — VOTING
# =========================================================

def to_word(subclass):
    return "_".join(subclass.split("_")[:-1]) if "_" in subclass else subclass

def probability_fallback(top_n_predictions, candidates=None):

    word_sums = defaultdict(float)
    word_counts = defaultdict(int)
    confident_found = False

    for preds in top_n_predictions:
        word = to_word(preds[0]["class"])
        prob = preds[0]["prob"]
        if candidates is not None and word not in candidates:
            continue
        if prob >= 0.5:
            confident_found = True
            word_sums[word] += prob
            word_counts[word] += 1

    if not confident_found:
        word_sums.clear()
        word_counts.clear()
        for preds in top_n_predictions:
            word = to_word(preds[0]["class"])
            prob = preds[0]["prob"]
            if candidates is not None and word not in candidates:
                continue
            word_sums[word] += prob
            word_counts[word] += 1

    if not word_sums:
        return "Could not identify", 0.0, 0, 0.0

    print("\nProbability Statistics")
    print("-" * 60)

    normalized_scores = {}

    for word in word_sums:

        normalized = word_sums[word] / word_counts[word]

        normalized_scores[word] = normalized

        print(
            f"{word:<15} "
            f"sum={word_sums[word]:.4f}  "
            f"frames={word_counts[word]:2d}  "
            f"normalized={normalized:.4f}"
        )

    # --------------------------------------------------
    # Find highest frame count
    # --------------------------------------------------

    max_count = max(word_counts.values())

    top_count_words = [
        word
        for word in word_counts
        if word_counts[word] == max_count
    ]

    print("\nWords with Highest Frame Count")
    print("-" * 40)

    for word in top_count_words:
        print(
            f"{word:<15} "
            f"frames={word_counts[word]} "
            f"sum={word_sums[word]:.4f}"
        )

    print("\nNormalized Scores (Top-Count Words)")
    print("-" * 40)

    best_normalized = -1

    for word in top_count_words:

        score = normalized_scores[word]

        print(f"{word:<15} {score:.4f}")

        best_normalized = max(
            best_normalized,
            score
        )

    winners = [
        word
        for word in top_count_words
        if abs(normalized_scores[word] - best_normalized) < 1e-8
    ]

    # --------------------------------------------------
    # Multiple winners
    # --------------------------------------------------

    if len(winners) > 1:

        print(
            "\n[Fallback] Multiple normalized-probability winners:"
        )

        for w in winners:
            print(
                f"{w} ({normalized_scores[w]:.4f})"
            )

        winner = " / ".join(sorted(winners))

    else:

        winner = winners[0]

        print(
            f"\n[Fallback] Using highest normalized probability "
            f"among top-count words: {winner}"
        )

    return (
        winner,
        normalized_scores[winners[0]],
        word_counts[winners[0]],
        word_sums[winners[0]]
    )

def decide_word_majority(pred_names):
    if not pred_names:
        return "Could not identify"

    word_votes   = [to_word(s) for s in pred_names]
    unique_sub   = set(pred_names)
    unique_words = set(word_votes)

    if len(unique_sub) == 1:
        return "Could not identify"

    if len(unique_sub) in (2, 3) and len(unique_words) == 1:
        return unique_words.pop()

    word_counter = Counter(word_votes)
    max_votes = max(word_counter.values())

    majority_words = [
        word
        for word, count in word_counter.items()
        if count == max_votes
    ]

    if len(majority_words) == 1:
        return majority_words[0]

    return majority_words


# =========================================================
# FULL PIPELINE
# =========================================================

def run(video_path, output_dir=None):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_dir is None:
        #output_dir = os.path.join(os.path.dirname(video_path),
        #                          f"pipeline_out_{video_name}")
        output_dir = os.path.join(os.path.dirname(video_path),
                                  f"{video_name}")
    os.makedirs(output_dir, exist_ok=True)

    skeletal_dir        = os.path.join(output_dir, "skeletal_images")
    original_frames_dir = os.path.join(output_dir, "original_keyframes")
    json_path           = os.path.join(output_dir, "keypoints.json")

    # Stage 1
    print("\n" + "="*60)
    print("STAGE 1 : KEYFRAME EXTRACTION")
    print("="*60)
    kf_model, device = load_keyframe_model(KEYFRAME_MODEL_PATH)
    keyframes        = extract_keyframes(video_path, kf_model, device)

    if not keyframes:
        print("No keyframes extracted. Aborting.")
        return

    # Stage 2
    print("\n" + "="*60)
    print("STAGE 2 : DWPose SKELETONISATION (NORMALIZED)")
    print("="*60)
    original_dir = os.getcwd()
    os.chdir(r"C:\Users\navne\OneDrive\Desktop\DWPose-onnx\ControlNet-v1-1-nightly")
    try:
        skeletal_paths, kp_records = run_dwpose(
            keyframes, skeletal_dir, original_frames_dir, json_path
        )
    finally:
        os.chdir(original_dir)

    # Stage 3
    print("\n" + "="*60)
    print("STAGE 3 : FRAME CLASSIFICATION")
    print("="*60)
    scaler, to_drop, le = load_scaler_and_le()

    keras_model = load_classifier_model(CLASSIFIER_MODEL_PATH)

    pred_names, top_n_predictions = classify_frames(
        keras_model, scaler, to_drop, le, skeletal_paths, kp_records
    )

    # Stage 4
    print("\n" + "="*60)
    print("STAGE 4 : VOTING")
    print("="*60)

    confident_pred_names = [
        name for name, preds in zip(pred_names, top_n_predictions)
        if preds[0]['prob'] >= 0.5
    ]

    print("\nPredictions:")
    print(pred_names)

    print("\nUnique subclasses:")
    print(set(pred_names))

    print("\nWord counts:")
    print(Counter(to_word(x) for x in pred_names))

    # --- majority vote: confident frames if available, else all frames ---
    vote_pred_names = confident_pred_names if confident_pred_names else pred_names
    maj_result = decide_word_majority(vote_pred_names)

    if isinstance(maj_result, list):
        # tie between multiple words -> break tie via probability_fallback,
        # restricted to just the tied candidates
        winner_word, winner_norm, winner_frames, winner_sum = \
            probability_fallback(top_n_predictions, candidates=maj_result)

    elif maj_result == "Could not identify":
        # majority vote inconclusive -> fall back to probability winner
        # (confident frames if available, else all frames — handled inside)
        winner_word, winner_norm, winner_frames, winner_sum = \
            probability_fallback(top_n_predictions)

    else:
        winner_word = maj_result

        # normalized probability for this word: confident frames if
        # available, else all frames
        def _stats_for_word(word, require_confident):
            total_prob, total_frames = 0.0, 0
            for preds in top_n_predictions:
                w = to_word(preds[0]["class"])
                p = preds[0]["prob"]
                if require_confident and p < 0.5:
                    continue
                if w == word:
                    total_prob += p
                    total_frames += 1
            return total_prob, total_frames

        winner_sum, winner_frames = _stats_for_word(winner_word, require_confident=True)
        if winner_frames == 0:
            # no confident frames for this word -> use all frames instead
            winner_sum, winner_frames = _stats_for_word(winner_word, require_confident=False)

        winner_norm = winner_sum / winner_frames if winner_frames > 0 else 0.0

    confidence_percent = winner_norm * 100

    print("\nFrame-level Predictions (score >= 0.5):\n")
    for path, preds in zip(skeletal_paths, top_n_predictions):
        if preds[0]['prob'] >= 0.5:
            print(
                f"{os.path.basename(path)}    "
                f"{preds[0]['class']} "
                f"({preds[0]['prob']:.4f})"
            )

    print("\nNormalized Probability")
    print("-" * 40)
    print(f"Word         : {winner_word}")
    print(f"Frames       : {winner_frames}")
    print(f"Sum Prob     : {winner_sum:.4f}")
    print(f"Normalized   : {winner_norm:.4f}")

    print(f"\n{'='*60}")
    print(f"Word        : {winner_word}")
    print(f"Confidence  : {confidence_percent:.2f}%")
    print(f"{'='*60}\n")

    # Save summary
    summary_path = os.path.join(output_dir, "prediction_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Video          : {video_path}\n")
        f.write(f"KF threshold   : >= {KF_THRESHOLD}\n")
        f.write(f"Keyframes used : {len(keyframes)}\n\n")

        f.write("Frame predictions:\n")
        for path, preds in zip(skeletal_paths, top_n_predictions):
            if preds[0]['prob'] >= 0.5:
                f.write(
                    f"{os.path.basename(path)}    "
                    f"{preds[0]['class']} "
                    f"({preds[0]['prob']:.4f})\n"
                )

        f.write("\n")
        f.write(f"Final Word        : {winner_word}\n")
        f.write(f"Frame Count       : {winner_frames}\n")
        f.write(f"Probability Sum   : {winner_sum:.4f}\n")
        f.write(f"Normalized Prob   : {winner_norm:.4f}\n")
        f.write(f"Confidence (%)    : {confidence_percent:.2f}\n")

    print(f"Summary saved to:\n  {summary_path}")

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Kathakali gesture word from a raw video."
    )
    parser.add_argument("--video", required=True,
        help="Path to the input .mp4 video file.")
    parser.add_argument("--output_dir", default=None,
        help="Directory to save skeletal images, keypoints JSON, and summary.")
    args = parser.parse_args()

    run(video_path=args.video, output_dir=args.output_dir)