import os
import re
import csv

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from predict_from_video import run

VIDEOS_ROOT = r"C:\Users\navne\Pictures\Kathakali\Videos_Cropped"

OUTPUT_ROOT = r"C:\Users\navne\Pictures\Kathakali\Test_Results_9_10_11"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

video_pattern = re.compile(
    r".*_(9|10)\.(mp4|avi|mov|mkv)$",
    re.IGNORECASE
)

all_videos = []

CSV_PATH = os.path.join(OUTPUT_ROOT, "results.csv")
REPORT_PATH = os.path.join(OUTPUT_ROOT, "evaluation_report.txt")

# --------------------------------------------------
# Find videos 9, 10
# --------------------------------------------------

for word_folder in sorted(os.listdir(VIDEOS_ROOT)):

    word_path = os.path.join(VIDEOS_ROOT, word_folder)

    if not os.path.isdir(word_path):
        continue

    for file in sorted(os.listdir(word_path)):

        if video_pattern.match(file):

            video_path = os.path.join(word_path, file)

            all_videos.append(video_path)

# --------------------------------------------------
# Create CSV
# --------------------------------------------------

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "video",
        "ground_truth",
        "prediction",
        "confidence",
        "correct"
    ])

print(f"\nFound {len(all_videos)} test videos\n")

# --------------------------------------------------
# Metrics tracking
# --------------------------------------------------

num_correct = 0

y_true = []
y_pred = []

# --------------------------------------------------
# Run pipeline
# --------------------------------------------------

for idx, video_path in enumerate(all_videos, start=1):

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    ground_truth = os.path.basename(
        os.path.dirname(video_path)
    )

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

        if correct:
            num_correct += 1

        y_true.append(ground_truth)
        y_pred.append(prediction)

        # --------------------------------------------------
        # Write row
        # --------------------------------------------------

        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                video_name,
                ground_truth,
                prediction,
                confidence,
                "YES" if correct else "NO"
            ])

        print(
            f"GT: {ground_truth} | "
            f"Pred: {prediction} | "
            f"{'YES' if correct else 'NO'}"
        )

    except Exception as e:

        print(f"FAILED: {video_name}")
        print(e)

# --------------------------------------------------
# Final Metrics
# --------------------------------------------------

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

num_videos = len(y_true)
num_wrong = num_videos - num_correct

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"Total Videos : {num_videos}")
print(f"Correct      : {num_correct}")
print(f"Wrong        : {num_wrong}")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")

print("\nCLASSIFICATION REPORT")
print("-" * 80)

report = classification_report(
    y_true,
    y_pred,
    digits=4,
    zero_division=0
)

print(report)

# --------------------------------------------------
# Save report
# --------------------------------------------------

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("FINAL RESULTS\n")
    f.write("=" * 80 + "\n")

    f.write(f"Total Videos : {num_videos}\n")
    f.write(f"Correct      : {num_correct}\n")
    f.write(f"Wrong        : {num_wrong}\n")
    f.write(f"Accuracy     : {accuracy:.4f}\n")
    f.write(f"Precision    : {precision:.4f}\n")
    f.write(f"Recall       : {recall:.4f}\n")
    f.write(f"F1 Score     : {f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 80 + "\n")
    f.write(report)

print("\nResults CSV:")
print(CSV_PATH)

print("\nEvaluation Report:")
print(REPORT_PATH)

print("\nFinished.")