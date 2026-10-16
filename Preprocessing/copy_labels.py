import pandas as pd


OLD_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\v1\dwpose_keypoints_old.csv"
NEW_CSV = r"C:\Users\navne\OneDrive\Desktop\IIITK\Research\Video_Classification\Preprocessing\dwpose_keypoints.csv"


if __name__ == "__main__":

    old_df = pd.read_csv(OLD_CSV)
    new_df = pd.read_csv(NEW_CSV)

    # Build lookup: image_path -> label from old CSV
    label_lookup = dict(zip(old_df["image_path"], old_df["label"]))

    before_unknown = (new_df["label"] == "UNKNOWN").sum()

    # Copy label from old CSV where image_path matches
    new_df["label"] = new_df["image_path"].map(label_lookup).fillna(new_df["label"])

    after_unknown = (new_df["label"] == "UNKNOWN").sum()

    new_df.to_csv(NEW_CSV, index=False)

    print(f"Total rows         : {len(new_df)}")
    print(f"Labels copied      : {before_unknown - after_unknown}")
    print(f"Still UNKNOWN      : {after_unknown}")
    print(f"Saved to           : {NEW_CSV}")
