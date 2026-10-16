import os
import re
import random

from predict_from_video import run

VIDEOS_ROOT = r"C:\Users\navne\Pictures\Kathakali\New_videos"

OUTPUT_ROOT = r"C:\Users\navne\Pictures\Kathakali\Pipelines"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

REPORT_PATH = r"C:\Users\navne\Pictures\Kathakali\Pipelines\results_new_videos.csv"

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# --------------------------------------------------
# Filenames look like "Aakasham_2_1.mp4", "Aana_3_5.mp4"
# Ground truth label is everything before the trailing
# "_<num>_<num>" part.
# --------------------------------------------------

LABEL_PATTERN = re.compile(r"^(.*)_\d+_\d+$")


def extract_ground_truth(video_name):

    match = LABEL_PATTERN.match(video_name)

    if match:
        return match.group(1).strip()

    return video_name.strip()

# --------------------------------------------------
# Group key is "word_person", e.g. "aakasham_2" from
# "aakasham_2_1", "aakasham_2_2", ... "aakasham_2_5".
# One video is picked at random per group.
# --------------------------------------------------

GROUP_PATTERN = re.compile(r"^(.*)_\d+$")


def extract_group_key(video_name):

    match = GROUP_PATTERN.match(video_name)

    if match:
        return match.group(1).strip()

    return video_name.strip()

# --------------------------------------------------
# Find all videos directly inside VIDEOS_ROOT, group by
# "word_person" and randomly pick one video per group
# --------------------------------------------------

groups = {}

for file in sorted(os.listdir(VIDEOS_ROOT)):

    file_path = os.path.join(VIDEOS_ROOT, file)

    if not os.path.isfile(file_path):
        continue

    if not file.lower().endswith(VIDEO_EXTENSIONS):
        continue

    if file.lower().startswith("z"):
        continue

    video_name = os.path.splitext(file)[0]

    group_key = extract_group_key(video_name)

    groups.setdefault(group_key, []).append(file_path)

all_videos = [
    random.choice(videos)
    for group_key, videos in sorted(groups.items())
]

print(f"\nFound {len(groups)} groups, selected {len(all_videos)} videos\n")

# --------------------------------------------------
# Accuracy tracking
# --------------------------------------------------

num_correct = 0
num_evaluated = 0

# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

for idx, video_path in enumerate(all_videos, start=1):

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    ground_truth = extract_ground_truth(video_name)

    output_dir = os.path.join(
        OUTPUT_ROOT,
        video_name
    )

    print("\n" + "=" * 80)
    print(f"[{idx}/{len(all_videos)}] {video_name}")
    print("=" * 80)

    try:

        run(
            video_path=video_path,
            output_dir=output_dir
        )

        # --------------------------------------------------
        # Parse prediction_summary.txt
        # --------------------------------------------------

        summary_path = os.path.join(
            output_dir,
            "prediction_summary.txt"
        )

        prediction = None
        confidence = None

        if os.path.exists(summary_path):

            with open(summary_path, "r", encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if "Final Word" in line:

                        prediction = line.split(":", 1)[1].strip()

                    elif "Confidence" in line:

                        confidence = line.split(":", 1)[1].strip()

        # --------------------------------------------------
        # Safety fallback
        # --------------------------------------------------

        if prediction is None:
            prediction = "UNKNOWN"

        if confidence is None:
            confidence = "0"

        # --------------------------------------------------
        # Correct?
        # --------------------------------------------------

        correct = (
            prediction.strip().lower()
            ==
            ground_truth.strip().lower()
        )

        num_evaluated += 1

        if correct:
            num_correct += 1

        print(
            f"GT: {ground_truth} | "
            f"Pred: {prediction} | "
            f"{'YES' if correct else 'NO'}"
        )

    except Exception as e:

        print(f"FAILED: {video_name}")
        print(e)

# --------------------------------------------------
# Final Accuracy
# --------------------------------------------------

accuracy = (num_correct / num_evaluated) if num_evaluated else 0.0

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(f"Total Videos : {num_evaluated}")
print(f"Correct      : {num_correct}")
print(f"Wrong        : {num_evaluated - num_correct}")
print(f"Accuracy     : {accuracy:.4f}")

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("FINAL RESULTS\n")
    f.write("=" * 80 + "\n")

    f.write(f"Total Videos : {num_evaluated}\n")
    f.write(f"Correct      : {num_correct}\n")
    f.write(f"Wrong        : {num_evaluated - num_correct}\n")
    f.write(f"Accuracy     : {accuracy:.4f}\n")

print("\nResults file:")
print(REPORT_PATH)

print("\nFinished.")