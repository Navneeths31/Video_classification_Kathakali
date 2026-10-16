import os
import cv2
import pandas as pd
import numpy as np

from annotator.dwpose import draw_pose
from annotator.dwpose.wholebody import Wholebody


# ==================================================
# INPUT / OUTPUT PATHS
# ==================================================

#INPUT_ROOT = r"C:\Users\navne\Pictures\Kathakali\Mudrapedia\Original_keyframes"
INPUT_ROOT = r"C:\Users\navne\Pictures\Kathakali\Uniform_Images"

#OUTPUT_SKELETAL_ROOT = r"C:\Users\navne\Pictures\Kathakali\Mudrapedia\Skeletal_keyframes"
OUTPUT_SKELETAL_ROOT = r"C:\Users\navne\Pictures\Kathakali\Uniform_Skeletal_Images_2"

#OUTPUT_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_keypoints_mudrapedia.csv"
OUTPUT_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_keypoints_2.csv"

FAILED_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_failed_images.csv"

#KEYFRAME_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\keyframes_probabilities_mudrapedia.csv"
KEYFRAME_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\keyframes_probabilities_2.csv"

# CSV #1: normalized limb length + angle features
LIMB_FEATURES_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_limb_features_2.csv"

# CSV #2: reconstructed (normalized) keypoint coordinates
RECONSTRUCTED_KP_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_reconstructed_keypoints_2.csv"

limb_csv_initialized = False


# ==================================================
# IMAGE EXTENSIONS
# ==================================================

def is_image_file(filename):
    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )
    return filename.lower().endswith(valid_extensions)


# ==================================================
# NORMALIZATION HELPERS
# ==================================================

def compute_limb(p1, p2, shoulder_width):
    """
    Given two 2D keypoints p1, p2 (already neck-translated, NOT yet
    shoulder-width-normalised), return:
        normalized_length  – Euclidean distance / shoulder_width
        angle              – arctan2(dy, dx) in radians  (unchanged)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx * dx + dy * dy)
    normalized_length = length / shoulder_width if shoulder_width > 0 else 0.0
    angle = np.arctan2(dy, dx)
    return normalized_length, angle


def reconstruct_point(parent, norm_length, angle):
    """
    Reconstruct a child keypoint from its parent using the stored
    normalised length and angle. Lengths are already normalised by
    shoulder width, so no scaling back is applied.
    """
    x = parent[0] + norm_length * np.cos(angle)
    y = parent[1] + norm_length * np.sin(angle)
    return np.array([x, y], dtype=np.float32)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # ----------------------------------------------
    # LOAD DWPose
    # ----------------------------------------------
    pose_estimation = Wholebody()

    # ----------------------------------------------
    # LOAD KEYFRAME CSV
    # ----------------------------------------------
    keyframe_df = pd.read_csv(KEYFRAME_CSV)

    # ----------------------------------------------
    # CREATE LOOKUP
    # (video_name, frame_id) -> assigned_keyframe
    # ----------------------------------------------
    keyframe_lookup = {}
    for _, row in keyframe_df.iterrows():
        frame_id = str(row["frame_id"]).strip()
        key = (str(row["video_name"]), frame_id)
        keyframe_lookup[key] = row["assigned_keyframe"]

    # ----------------------------------------------
    # KEEP ONLY: upper body + hands
    # ----------------------------------------------
    keep_indices = [
        # upper body
        0, 1, 2, 3, 4, 5, 6, 7,
        # left hand
        *range(92, 113),
        # right hand
        *range(113, 134)
    ]

    # ----------------------------------------------
    # KEYPOINT NAMES  (50 keypoints total)
    # ----------------------------------------------
    keypoint_names = [
        # upper body (indices 0-7)
        "nose", "neck",
        "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist",
        # left hand  (indices 8-28)
        *[f"l_hand_{i}" for i in range(21)],
        # right hand (indices 29-49)
        *[f"r_hand_{i}" for i in range(21)]
    ]

    kp_idx = {name: i for i, name in enumerate(keypoint_names)}

    # ----------------------------------------------
    # LIMB CHAIN DEFINITION
    # Each tuple: (feature_name, parent_kp_name, child_kp_name)
    # The order matters for reconstruction – parent must be
    # reconstructed before the child.
    # ----------------------------------------------
    limb_chain = [
        # shoulders from neck
        ("neck_r_shoulder",  "neck",        "r_shoulder"),
        ("neck_l_shoulder",  "neck",        "l_shoulder"),

        # right arm
        ("r_upper_arm",      "r_shoulder",  "r_elbow"),
        ("r_forearm",        "r_elbow",     "r_wrist"),
        ("r_wrist_hand",     "r_wrist",     "r_hand_0"),

        # left arm
        ("l_upper_arm",      "l_shoulder",  "l_elbow"),
        ("l_forearm",        "l_elbow",     "l_wrist"),
        ("l_wrist_hand",     "l_wrist",     "l_hand_0"),

        # left hand fingers
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

        # right hand fingers
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

    # ----------------------------------------------
    # COUNTERS
    # ----------------------------------------------
    total_processed = 0
    total_failed = 0

    csv_initialized          = False
    failed_csv_initialized   = False
    limb_csv_initialized     = False
    recon_csv_initialized    = False

    # ==============================================
    # WALK THROUGH ALL IMAGES
    # ==============================================
    for root, dirs, files in os.walk(INPUT_ROOT):

        for file in files:

            if not is_image_file(file):
                continue

            image_path = os.path.join(root, file)

            print("\n===================================")
            print(f"Processing: {image_path}")

            image = cv2.imread(image_path)

            # ------------------------------------------
            # IMAGE READ FAILURE
            # ------------------------------------------
            if image is None:
                print("Failed to read image")
                failed_row = pd.DataFrame([{"image_path": image_path, "reason": "Failed to read image"}])
                failed_row.to_csv(
                    FAILED_CSV,
                    mode='a',
                    header=not failed_csv_initialized and not os.path.exists(FAILED_CSV),
                    index=False
                )
                failed_csv_initialized = True
                total_failed += 1
                continue

            # Resize every image to 512x512
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

            try:

                H, W, _ = image.shape

                # ======================================
                # STAGE 1: EXTRACT KEYPOINTS
                # ======================================
                keypoints, scores = pose_estimation(image)

                full_scores = scores.copy()

                if len(keypoints) == 0:
                    raise Exception("No person detected")

                keypoints = keypoints[:, keep_indices, :]
                scores    = scores[:, keep_indices]

                # Translate so neck is at (0, 0)
                neck_x = keypoints[0, 1, 0]
                neck_y = keypoints[0, 1, 1]

                if neck_x == 0 and neck_y == 0:
                    raise Exception("Neck not detected, skipping")

                translation = np.array([neck_x, neck_y])
                keypoints[..., :2] -= translation

                # Scale by image dimensions
                keypoints[..., 0] /= float(W)
                keypoints[..., 1] /= float(H)

                # flattened_keypoints shape: (50, 2)
                flattened_keypoints = keypoints[0]

                # ======================================
                # STAGE 2: COMPUTE SHOULDER WIDTH
                # ======================================
                r_shoulder_pt = flattened_keypoints[kp_idx["r_shoulder"]]
                l_shoulder_pt = flattened_keypoints[kp_idx["l_shoulder"]]
                shoulder_width = np.linalg.norm(r_shoulder_pt - l_shoulder_pt)

                if shoulder_width < 1e-6:
                    raise Exception("Shoulder width too small, cannot normalise")

                print(f"\nShoulder width: {shoulder_width:.4f}")

                # ======================================
                # STAGE 3 & 4: COMPUTE LIMB LENGTH + ANGLE
                # ======================================
                limb_features = {}

                for (feat_name, p1_name, p2_name) in limb_chain:
                    p1 = flattened_keypoints[kp_idx[p1_name]]
                    p2 = flattened_keypoints[kp_idx[p2_name]]
                    norm_len, angle = compute_limb(p1, p2, shoulder_width)
                    limb_features[feat_name] = (norm_len, angle)

                # Nose relative to neck
                nose_pt      = flattened_keypoints[kp_idx["nose"]]
                neck_pt_orig = flattened_keypoints[kp_idx["neck"]]
                nose_norm_len, nose_angle = compute_limb(neck_pt_orig, nose_pt, shoulder_width)

                # ======================================
                # STAGE 5: RECONSTRUCT SKELETON
                # Start from neck = (0, 0)
                # ======================================
                reconstructed = {}
                reconstructed["neck"] = np.array([0.0, 0.0], dtype=np.float32)

                # Reconstruct nose from neck
                reconstructed["nose"] = reconstruct_point(
                    reconstructed["neck"], nose_norm_len, nose_angle
                )

                # Walk the limb chain in order (parent always available before child)
                for (feat_name, p1_name, p2_name) in limb_chain:
                    norm_len, angle = limb_features[feat_name]
                    reconstructed[p2_name] = reconstruct_point(
                        reconstructed[p1_name], norm_len, angle
                    )

                # ======================================
                # STAGE 6: BUILD RECONSTRUCTED KEYPOINT ARRAY
                # Same order as keypoint_names
                # ======================================
                reconstructed_keypoints = np.zeros((len(keypoint_names), 2), dtype=np.float32)
                for i, name in enumerate(keypoint_names):
                    if name in reconstructed:
                        reconstructed_keypoints[i] = reconstructed[name]
                    else:
                        reconstructed_keypoints[i] = flattened_keypoints[i]

                # ======================================
                # GET LABEL FROM KEYFRAME CSV
                # ======================================
                relative_path = os.path.relpath(root, INPUT_ROOT)
                folder_parts  = relative_path.split(os.sep)
                video_name    = folder_parts[-1]
                filename_without_ext = os.path.splitext(file)[0]

                lookup_key = (video_name, filename_without_ext)
                label = keyframe_lookup.get(lookup_key, "UNKNOWN")

                # ======================================
                # STAGE 7a: SAVE CSV #1 (limb features)
                # ======================================
                limb_row = {
                    "image_path": image_path,
                    "label": label
                }

                print("\nLimb Features (normalized length | angle rad)")
                print("-" * 60)

                for feat_name, (norm_len, angle) in limb_features.items():
                    limb_row[f"{feat_name}_length"] = norm_len
                    limb_row[f"{feat_name}_angle"]  = angle
                    print(f"  {feat_name:<25}  len={norm_len:.4f}  angle={angle:.4f}")

                limb_row["nose_length"] = nose_norm_len
                limb_row["nose_angle"]  = nose_angle

                limb_df = pd.DataFrame([limb_row])
                limb_df.to_csv(
                    LIMB_FEATURES_CSV,
                    mode='a',
                    header=not limb_csv_initialized and not os.path.exists(LIMB_FEATURES_CSV),
                    index=False
                )
                limb_csv_initialized = True

                # ======================================
                # STAGE 7b: SAVE CSV #2 (reconstructed coords)
                # ======================================
                recon_row = {
                    "image_path": image_path,
                    "label": label
                }
                for i, name in enumerate(keypoint_names):
                    recon_row[f"{name}_x"] = reconstructed_keypoints[i][0]
                    recon_row[f"{name}_y"] = reconstructed_keypoints[i][1]

                recon_df = pd.DataFrame([recon_row])
                recon_df.to_csv(
                    RECONSTRUCTED_KP_CSV,
                    mode='a',
                    header=not recon_csv_initialized and not os.path.exists(RECONSTRUCTED_KP_CSV),
                    index=False
                )
                recon_csv_initialized = True

                # ======================================
                # STAGE 8: DRAW SKELETON FROM
                # RECONSTRUCTED KEYPOINTS
                # ======================================

                # Build full 134-point candidate array defaulting to -1
                full_candidate = np.full((1, 134, 2), -1, dtype=np.float32)

                # Insert reconstructed keypoints at their original DWPose indices
                for out_idx, orig_idx in enumerate(keep_indices):
                    full_candidate[0, orig_idx, :] = reconstructed_keypoints[out_idx]

                # Shift only the filled slots so neck (0,0) maps to canvas centre (0.5, 0.5)
                for orig_idx in keep_indices:
                    full_candidate[0, orig_idx, :] += 0.5

                # Reconstruct body candidate + subset
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

                print(body)
                skeletal_image = draw_pose(pose, H, W)

                # ======================================
                # SAVE SKELETAL IMAGE
                # ======================================
                output_dir = os.path.join(OUTPUT_SKELETAL_ROOT, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_image_path = os.path.join(output_dir, file)
                cv2.imwrite(output_image_path, skeletal_image)

                print("Skeletal image saved")

                # ======================================
                # ALSO SAVE ORIGINAL-COORDS CSV
                # ======================================
                orig_row = {
                    "image_path": image_path,
                    "skeletal_image_path": output_image_path,
                    "label": label
                }
                for i, name in enumerate(keypoint_names):
                    orig_row[f"{name}_x"] = flattened_keypoints[i][0]
                    orig_row[f"{name}_y"] = flattened_keypoints[i][1]

                df = pd.DataFrame([orig_row])
                df.to_csv(
                    OUTPUT_CSV,
                    mode='a',
                    header=not csv_initialized and not os.path.exists(OUTPUT_CSV),
                    index=False
                )
                csv_initialized = True

                total_processed += 1
                print("Keypoints extracted, skeleton reconstructed, CSVs updated")

            except Exception as e:

                print("FAILED")
                print(f"Reason: {e}")

                failed_row = pd.DataFrame([{"image_path": image_path, "reason": str(e)}])
                failed_row.to_csv(
                    FAILED_CSV,
                    mode='a',
                    header=not failed_csv_initialized and not os.path.exists(FAILED_CSV),
                    index=False
                )
                failed_csv_initialized = True
                total_failed += 1


    # ==============================================
    # FINAL SUMMARY
    # ==============================================
    print("\n===================================")
    print("PROCESS COMPLETE")
    print("===================================")
    print(f"Successful: {total_processed}")
    print(f"Failed:     {total_failed}")
    print(f"\nOriginal keypoints CSV:      {OUTPUT_CSV}")
    print(f"Limb features CSV:           {LIMB_FEATURES_CSV}")
    print(f"Reconstructed keypoints CSV: {RECONSTRUCTED_KP_CSV}")
    print(f"Failed images CSV:           {FAILED_CSV}")
    print(f"Skeletal images:             {OUTPUT_SKELETAL_ROOT}")
