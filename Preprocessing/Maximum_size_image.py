import os
from PIL import Image

def find_max_size_images(directory):
    max_width = 0
    max_height = 0
    max_width_file = ""
    max_height_file = ""
    max_area = 0
    max_area_file = ""
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico'}
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in image_extensions:
                filepath = os.path.join(root, filename)
                try:
                    with Image.open(filepath) as img:
                        w, h = img.size
                        if w > max_width:
                            max_width = w
                            max_width_file = filepath
                        if h > max_height:
                            max_height = h
                            max_height_file = filepath
                        if w * h > max_area:
                            max_area = w * h
                            max_area_file = filepath
                except Exception:
                    pass
    
    print(f"Max Width : {max_width}px  →  {max_width_file}")
    print(f"Max Height: {max_height}px  →  {max_height_file}")
    print(f"Max Area  : {max_area}px²  →  {max_area_file}")

directory = r"C:\Users\navne\Pictures\Kathakali\Uniform_Images"
find_max_size_images(directory)