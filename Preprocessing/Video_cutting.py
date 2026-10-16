import cv2

# =========================================================
# INPUTS
# =========================================================
INPUT_VIDEO = r"C:\Users\navne\Pictures\Kathakali\New_videos\Aana_4_5.mp4"
#INPUT_VIDEO = r"C:\Users\navne\OneDrive\Videos\Irikkuka_5_2.mp4"
OUTPUT_VIDEO = r"C:\Users\navne\OneDrive\Desktop\Aana_4_5_cropped.mp4"

# =========================================================
# CROP RECTANGLE
# =========================================================
# Top-left corner
x1 = 1600
#x1 = 600
y1 = 300
#y1 = 100

# Bottom-right corner
x2 = 2840
#x2 = 1400
y2 = 2060
#y2 = 980

# =========================================================
# LOAD VIDEO
# =========================================================
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise ValueError(f"Cannot open video:\n{INPUT_VIDEO}")

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video resolution: {video_width} x {video_height}")

# =========================================================
# VALIDATE CROP
# =========================================================
if not (0 <= x1 < x2 <= video_width):
    raise ValueError(
        f"Invalid x coordinates.\n"
        f"Video width = {video_width}\n"
        f"Received: x1={x1}, x2={x2}"
    )

if not (0 <= y1 < y2 <= video_height):
    raise ValueError(
        f"Invalid y coordinates.\n"
        f"Video height = {video_height}\n"
        f"Received: y1={y1}, y2={y2}"
    )

crop_width = x2 - x1
crop_height = y2 - y1

# =========================================================
# OUTPUT VIDEO
# =========================================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (crop_width, crop_height)
)

# =========================================================
# PROCESS FRAMES
# =========================================================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    cropped = frame[y1:y2, x1:x2]
    out.write(cropped)

cap.release()
out.release()

print(f"Saved cropped video to:\n{OUTPUT_VIDEO}")