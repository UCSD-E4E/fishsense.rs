import json
import os
import cv2

WIDTH = 40.14
HEIGHT = 30.16
SEGMENTED_DIR = "./data/segmented"
TRUTH_DIR = "./data/truth"

os.makedirs(TRUTH_DIR, exist_ok=True)

with open("./data/diff_output.json", "r") as f:
    diff_data = json.load(f)

with open("./data/cleaned_data.json", "r") as f:
    cleaned_data = json.load(f)

cleaned_by_id = {str(entry["id"]): entry for entry in cleaned_data}

def percent_to_pixel(p):
    return int(p["x"] * WIDTH), int(p["y"] * HEIGHT)

for item in diff_data:
    image_id = item["id"]
    numeric_id = image_id.split("_")[0]

    if numeric_id not in cleaned_by_id:
        print(f"No cleaned data match for {image_id}")
        continue

    cleaned_entry = cleaned_by_id[numeric_id]
    snout_px = percent_to_pixel(cleaned_entry["snout"])
    fork_px = percent_to_pixel(cleaned_entry["fork"])

    image_filename = f"{image_id}.jpeg"
    input_path = os.path.join(SEGMENTED_DIR, image_filename)
    output_path = os.path.join(TRUTH_DIR, image_filename)

    if not os.path.exists(input_path):
        print(f"Image file not found: {input_path}")
        continue

    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to load image: {input_path}")
        continue

    cv2.circle(img, snout_px, 10, (0, 255, 0), -1)
    cv2.putText(img, "snout", (snout_px[0] + 10, snout_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.circle(img, fork_px, 10, (0, 0, 255), -1)
    cv2.putText(img, "fork", (fork_px[0] + 10, fork_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(output_path, img)
    print(f"Annotated image saved: {output_path}")
