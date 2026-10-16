import pandas as pd

csv_path = r"C:\Users\navne\Pictures\Kathakali\dwpose_limb_lengths.csv"

df = pd.read_csv(csv_path)

# Columns that contain limb lengths
exclude_cols = ["image_path", "label"]
limb_cols = [c for c in df.columns if c not in exclude_cols]

# Largest value in each row
df["max_length"] = df[limb_cols].max(axis=1)

# Column name that contains the largest value
df["max_limb"] = df[limb_cols].idxmax(axis=1)

print(df[["image_path", "label", "max_limb", "max_length"]])

df[["image_path", "label", "max_limb", "max_length"]].to_csv(
    r"C:\Users\navne\Pictures\Kathakali\largest_limb_per_image.csv",
    index=False
)

print(df['max_limb'])