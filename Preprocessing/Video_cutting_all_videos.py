import cv2
import os
import glob

# =========================================================
# INPUTS
# =========================================================
INPUT_FOLDER = r"C:\Users\navne\Pictures\Kathakali\New_videos"
PIPELINE_FOLDER = r"C:\Users\navne\Pictures\Kathakali\Pipelines"
OUTPUT_FOLDER = r"C:\Users\navne\Pictures\Kathakali\New_videos_cropped"

# =========================================================
# CROP RECTANGLES BY RESOLUTION
# =========================================================
CROP_SETTINGS = {
    (3840, 2160): {"x1": 1600, "y1": 300, "x2": 2840, "y2": 2060},
    (1920, 1080): {"x1": 600,  "y1": 100, "x2": 1400, "y2": 980},
}

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# =========================================================
# GET SUBFOLDER NAMES FROM PIPELINE FOLDER
# =========================================================
pipeline_names = {
    name for name in os.listdir(PIPELINE_FOLDER)
    if os.path.isdir(os.path.join(PIPELINE_FOLDER, name))
}

print(f"Found {len(pipeline_names)} folder(s) in Pipeline folder to match against.")

# =========================================================
# SETUP
# =========================================================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

all_video_files = [
    f for f in glob.glob(os.path.join(INPUT_FOLDER, "*"))
    if f.lower().endswith(VIDEO_EXTENSIONS)
]

# Only keep videos whose base name matches a Pipeline subfolder name
video_files = [
    f for f in all_video_files
    if os.path.splitext(os.path.basename(f))[0] in pipeline_names
]

matched_names = {os.path.splitext(os.path.basename(f))[0] for f in video_files}
unmatched_pipeline_names = pipeline_names - matched_names

print(f"{len(video_files)} of {len(all_video_files)} video(s) in New_videos matched a Pipeline folder.")
if unmatched_pipeline_names:
    print(f"Warning: {len(unmatched_pipeline_names)} Pipeline folder(s) had no matching video in New_videos:")
    for name in sorted(unmatched_pipeline_names):
        print(f"  - {name}")

if not video_files:
    print("No matching videos found. Nothing to process.")

# =========================================================
# PROCESS EACH MATCHED VIDEO
# =========================================================
for INPUT_VIDEO in video_files:
    filename = os.path.basename(INPUT_VIDEO)
    name_no_ext, _ = os.path.splitext(filename)
    OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, f"{name_no_ext}_cropped.mp4")

    print(f"\nProcessing: {filename}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"  Skipping — cannot open video: {INPUT_VIDEO}")
        continue

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Resolution: {video_width} x {video_height}")

    crop = CROP_SETTINGS.get((video_width, video_height))
    if crop is None:
        print(f"  Skipping — no crop settings defined for resolution "
              f"{video_width}x{video_height}")
        cap.release()
        continue

    x1, y1, x2, y2 = crop["x1"], crop["y1"], crop["x2"], crop["y2"]

    # -----------------------------------------------------
    # VALIDATE CROP
    # -----------------------------------------------------
    if not (0 <= x1 < x2 <= video_width):
        print(f"  Skipping — invalid x coordinates for this resolution "
              f"(x1={x1}, x2={x2}, video_width={video_width})")
        cap.release()
        continue
    if not (0 <= y1 < y2 <= video_height):
        print(f"  Skipping — invalid y coordinates for this resolution "
              f"(y1={y1}, y2={y2}, video_height={video_height})")
        cap.release()
        continue

    crop_width = x2 - x1
    crop_height = y2 - y1

    # -----------------------------------------------------
    # OUTPUT VIDEO
    # -----------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (crop_width, crop_height)
    )

    # -----------------------------------------------------
    # PROCESS FRAMES
    # -----------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y1:y2, x1:x2]
        out.write(cropped)

    cap.release()
    out.release()
    print(f"  Saved to: {OUTPUT_VIDEO}")

print("\nDone processing all matched videos.")