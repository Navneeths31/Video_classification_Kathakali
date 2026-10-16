import pandas as pd

CSV_PATH    = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_keypoints_hpc.csv"
OUTPUT_PATH = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_keypoints_hpc_2.csv"

OLD_PREFIX = r"C:\Users\navne\Pictures\Kathakali\Uniform_Images"
NEW_PREFIX = "/home/navaneeths/Uniform_Images"

df = pd.read_csv(CSV_PATH)

print("BEFORE:", df["image_path"].iloc[0])

df["image_path"] = (
    df["image_path"]
    .str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)
    .str.replace("\\", "/", regex=False)
)

print("AFTER: ", df["image_path"].iloc[0])

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved to:", OUTPUT_PATH)