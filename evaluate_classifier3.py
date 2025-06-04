import csv
import subprocess
import os
import re
import tempfile
from PIL import Image

ZOOM_CROP_FACTOR = 0.75 

GROUND_TRUTH_CSV = "/Users/tanishqsingh/Documents/GitHub/fishsense.rs/ground_truth.csv"
IMAGE_DIRECTORIES = [
    "/Users/tanishqsingh/Downloads/2024.06.27.FishSense.CCFRP/iPhone",
    "/Users/tanishqsingh/Downloads/2024.06.27.FishSense.CCFRP/iPad",
]
FISHSENSE_PROJECT_DIR = os.getcwd()

def find_image_path(base_filename, directories):
    for directory in directories:
        path = os.path.join(directory, base_filename)
        if os.path.exists(path):
            return path
    return None

def preprocess_image_and_zoom(input_image_path, output_image_path, crop_factor):
    try:
        img = Image.open(input_image_path)
        original_width, original_height = img.size

        crop_width = int(original_width * crop_factor)
        crop_height = int(original_height * crop_factor)

        left = (original_width - crop_width) / 2
        top = (original_height - crop_height) / 2
        right = (original_width + crop_width) / 2
        bottom = (original_height + crop_height) / 2

        cropped_img = img.crop((int(left), int(top), int(right), int(bottom)))
        
        if cropped_img.mode in ('RGBA', 'P'):
            cropped_img = cropped_img.convert('RGB')
            
        cropped_img.save(output_image_path, "JPEG")
        print(f"  Successfully zoomed and saved to {output_image_path}")
        return True
    except FileNotFoundError:
        print(f"  Error during preprocessing: File not found at {input_image_path}")
        return False
    except Exception as e:
        print(f"  Error during image preprocessing for {input_image_path}: {e}")
        return False

def run_fishsense_classifier(image_path):
    if not image_path:
        return None

    command = ["cargo", "run", "--release", "--", image_path]
    
    try:
        process = subprocess.Popen(command, cwd=FISHSENSE_PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate(timeout=60) 

        if process.returncode != 0:
            print(f"  Error running FishSense for {os.path.basename(image_path)}:")
            print(f"  Stderr: {stderr.strip()}")
            return None

        match = re.search(r"Predicted class:\s*(.+)", stdout)
        if match:
            predicted_species = match.group(1).strip()
            return predicted_species
        elif "Classification: None" in stdout: 
            print(f"  FishSense returned None classification for {os.path.basename(image_path)}")
            return "None" 
        else:
            print(f"  Could not parse 'Predicted class:' or 'Classification: None' from FishSense output for {os.path.basename(image_path)}:")
            print(f"  Stdout:\n{stdout.strip()}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  Timeout running FishSense for {os.path.basename(image_path)}")
        return None
    except Exception as e:
        print(f"  An exception occurred while running FishSense for {os.path.basename(image_path)}: {e}")
        return None

def main():
    if not os.path.exists(GROUND_TRUTH_CSV):
        print(f"Error: Ground truth file '{GROUND_TRUTH_CSV}' not found.")
        return

    ground_truth_data = []
    try:
        with open(GROUND_TRUTH_CSV, mode='r', encoding='utf-8-sig') as f: 
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Error: Could not read field names from '{GROUND_TRUTH_CSV}'.")
                return
            
            if 'File Name' not in reader.fieldnames or 'True Species' not in reader.fieldnames:
                print("Error: CSV file must contain 'File Name' and 'True Species' columns.")
                return

            for row in reader:
                ground_truth_data.append(row)
    except Exception as e:
        print(f"Error reading or parsing CSV file '{GROUND_TRUTH_CSV}': {e}")
        return

    if not ground_truth_data:
        print(f"No data rows found in {GROUND_TRUTH_CSV}.")
        return

    correct_predictions = 0
    total_images_processed = 0 
    results_log = [] 

    print(f"\nStarting FishSense classification accuracy evaluation with {len(ground_truth_data)} entries from CSV...\n")

    for i, entry in enumerate(ground_truth_data):
        path_from_csv_original = entry.get('File Name', '')
        path_from_csv = path_from_csv_original.strip()
        
        true_species = entry.get('True Species', '').strip()

        if not path_from_csv or not true_species:
            print(f"Skipping invalid row #{i+1}: Original entry: {entry}")
            results_log.append({
                "filename_csv": path_from_csv_original,
                "true_species": entry.get('True Species', ''),
                "predicted_species": "N/A (Skipped invalid row)",
                "is_correct": False
            })
            continue
        
        print(f"Processing CSV row #{i+1}: '{path_from_csv}' (True species: {true_species})...")
        
        image_full_path = None
        if os.path.isabs(path_from_csv): 
            if os.path.exists(path_from_csv):
                image_full_path = path_from_csv
                print(f"  Path from CSV is absolute and exists: '{image_full_path}'")
            else:
                print(f"  Absolute path from CSV not found: '{path_from_csv}'. Trying to find base name...")
                base_name = os.path.basename(path_from_csv)
                image_full_path = find_image_path(base_name, IMAGE_DIRECTORIES)
                if image_full_path:
                    print(f"  Found base name '{base_name}' at: {image_full_path}")
        else: 
            print(f"  Path '{path_from_csv}' from CSV is relative. Searching in IMAGE_DIRECTORIES...")
            image_full_path = find_image_path(path_from_csv, IMAGE_DIRECTORIES)
            if image_full_path:
                 print(f"  Found at: {image_full_path}")

        if not image_full_path or not os.path.exists(image_full_path):
            actual_path_searched = image_full_path if image_full_path else path_from_csv
            print(f"  Image not found or path invalid: {actual_path_searched}")
            results_log.append({
                "filename_csv": path_from_csv_original,
                "true_species": entry.get('True Species', ''),
                "predicted_species": f"N/A (Image not found at '{actual_path_searched}')",
                "is_correct": False
            })
            continue
        
        temp_zoomed_image_path = None
        predicted_species = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_zoomed_image_path = tmp_file.name
            
            if preprocess_image_and_zoom(image_full_path, temp_zoomed_image_path, ZOOM_CROP_FACTOR):
                total_images_processed += 1
                predicted_species = run_fishsense_classifier(temp_zoomed_image_path)
            else:
                print(f"  Skipping FishSense classification for {os.path.basename(image_full_path)} due to preprocessing error.")
                results_log.append({
                    "filename_csv": path_from_csv_original,
                    "true_species": entry.get('True Species', ''),
                    "predicted_species": "N/A (Preprocessing failed)",
                    "is_correct": False
                })
                continue 

        finally:
            if temp_zoomed_image_path and os.path.exists(temp_zoomed_image_path):
                os.remove(temp_zoomed_image_path)
        
        if predicted_species is not None: 
            print(f"  Predicted species: {predicted_species}")
            is_correct = predicted_species.strip().lower() == true_species.strip().lower()
            if is_correct:
                correct_predictions += 1
                print("  Result: CORRECT")
            else:
                print("  Result: INCORRECT")
            results_log.append({
                "filename_csv": path_from_csv_original,
                "true_species": entry.get('True Species', ''),
                "predicted_species": predicted_species,
                "is_correct": is_correct
            })
        elif total_images_processed > 0 and not any(r["filename_csv"] == path_from_csv_original and "Preprocessing failed" in r["predicted_species"] for r in results_log):
             print(f"  Failed to get prediction for {os.path.basename(image_full_path)} from FishSense.")
             results_log.append({
                "filename_csv": path_from_csv_original,
                "true_species": entry.get('True Species', ''),
                "predicted_species": "N/A (Prediction failed or None)",
                "is_correct": False
            })
        print("-" * 30)

    print("\n Evaluation Summary ")
    for log_entry in results_log:
        status_indicator = "CORRECT" if log_entry["is_correct"] else "INCORRECT"
        details = ""
        if not log_entry["is_correct"]:
            if "Skipped" in log_entry["predicted_species"]:
                 details = f" (Reason: Skipped invalid CSV row)"
            elif "Image not found" in log_entry["predicted_species"]:
                details = f" (Reason: Image not found)"
            elif "Preprocessing failed" in log_entry["predicted_species"]:
                details = f" (Reason: Image preprocessing failed)"
            elif "Prediction failed or None" in log_entry["predicted_species"]:
                details = f" (Reason: Prediction failed/None from FishSense)"
        
        print(f"File: {log_entry['filename_csv']}, True: {log_entry['true_species']}, Predicted: {log_entry['predicted_species']}, Status: {status_indicator}{details}")

    if total_images_processed > 0:
        accuracy = (correct_predictions / total_images_processed) * 100
        print(f"\nTotal images effectively processed: {total_images_processed} (out of {len(ground_truth_data)} CSV entries)")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print(f"No images were successfully processed out of {len(ground_truth_data)} CSV entries.")
        if len(ground_truth_data) > 0:
            print("Please check error messages")

if __name__ == "__main__":
    main()