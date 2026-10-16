import cv2

INPUT_VIDEO = r"C:\Users\navne\Pictures\Kathakali\Mudrapedia_videos\Paadam.mp4"
OUTPUT_VIDEO = r"C:\Users\navne\Pictures\Kathakali\Mudrapedia_videos\Paadam_flipped.mp4"

cap = cv2.VideoCapture(INPUT_VIDEO)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    out.write(flipped_frame)

cap.release()
out.release()

print(f"Saved flipped video to: {OUTPUT_VIDEO}")