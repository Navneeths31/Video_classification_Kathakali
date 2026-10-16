import os
from PIL import Image

# Input root folder
input_root = r"C:\Users\navne\Pictures\Kathakali\Uniform_Skeletal_Images"
# Output root folder
output_root = r"C:\Users\navne\Pictures\Kathakali\Uniform_Greyscale_Skeletal_Images"

# Traverse all folders and subfolders
for root, dirs, files in os.walk(input_root):
    # Relative path from input root
    rel_path = os.path.relpath(root, input_root)

    # Corresponding output folder
    output_dir = os.path.join(output_root, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)

            try:
                # Open image
                img = Image.open(input_path)

                # Convert to grayscale
                gray = img.convert("L")

                # Save to output folder
                gray.save(output_path)

                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

print("All images converted successfully.")