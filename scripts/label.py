import os
import json
from PIL import Image
import numpy as np

with open('./data/cleaned_data.json') as f:
    annotations_list = json.load(f)
    annotations = {str(entry["id"]): entry for entry in annotations_list}

# Path to segmented images
seg_path = './data/segmented'

# # Helper to check if a point is inside the fish mask
# def is_inside(image_array, x, y):
#     h, w = image_array.shape
#     x, y = int(round(x)), int(round(y))
#     return 0 <= x < w and 0 <= y < h and image_array[y, x] != 0

def is_inside(image_array, x, y, radius=10):
    h, w = image_array.shape
    x, y = int(round(x)), int(round(y))

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and image_array[ny, nx] != 0:
                return True
    return False
su = 0
prev = set()
for filename in os.listdir(seg_path):

    name, ext = os.path.splitext(filename)

    image_id, _ = name.split('_', 1)
    entry = annotations.get(image_id)

    snout = entry.get("snout")
    fork = entry.get("fork")

    # Load segmented image
    image_path = os.path.join(seg_path, filename)
    image = Image.open(image_path).convert("L")
    img_array = np.array(image)
    # print(snout)

    snout_inside = is_inside(img_array, snout["x"]/100*4014, snout["y"]/100*3016)
    fork_inside = is_inside(img_array, fork["x"]/100*4014, fork["y"]/100*3016)
    if image_id == "200140":
        print(image_id)
        print(snout["x"]/100*4014, snout["y"]/100*3016)
    if snout_inside or fork_inside:
        if image_id in prev:
            print("duplicate")
        prev.add(image_id)
        new_name = f"{name}_special{ext}"
        new_path = os.path.join(seg_path, new_name)
        # os.rename(image_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
        su += 1
print(su)