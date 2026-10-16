import os
import hashlib
from collections import defaultdict


# ==================================================
# INPUT PATH
# ==================================================

INPUT_ROOT = r"C:\Users\navne\Pictures\Kathakali\Uniform_Images_2"


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
# HASH FILE
# ==================================================

def hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # hash -> list of paths
    hash_map = defaultdict(list)

    total = 0

    print("Scanning images...")

    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            if not is_image_file(file):
                continue
            path = os.path.join(root, file)
            file_hash = hash_file(path)
            hash_map[file_hash].append(path)
            total += 1

    print(f"\nTotal images found : {total}")

    # Filter to only hashes with more than one path
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    duplicate_count = sum(len(paths) - 1 for paths in duplicates.values())

    print(f"Unique images      : {total - duplicate_count}")
    print(f"Duplicate images   : {duplicate_count}")
    print(f"Duplicate groups   : {len(duplicates)}")

    if duplicates:
        print("\n===================================")
        print("DUPLICATE GROUPS")
        print("===================================")
        for i, (h, paths) in enumerate(duplicates.items(), 1):
            print(f"\nGroup {i} (md5: {h})")
            for p in paths:
                print(f"  {p}")
    else:
        print("\nNo duplicates found.")
