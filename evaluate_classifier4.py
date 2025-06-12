import csv
import subprocess
import os
import re
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        # print(f"  Successfully zoomed and saved to {output_image_path}")
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
            # print(f"  Stdout:\n{stdout.strip()}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  Timeout running FishSense for {os.path.basename(image_path)}")
        return None
    except Exception as e:
        print(f"  An exception occurred while running FishSense for {os.path.basename(image_path)}: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, class_names, output_filename="confusion_matrix.png"):
    if not y_true or not y_pred:
        print("Not enough data to generate a confusion matrix.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Species")
    plt.xlabel("Predicted Species")
    
    # Adjust layout to prevent labels from being cut off
    num_classes = len(class_names)
    if num_classes > 10: # Heuristic for when to rotate
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename)
        print(f"\nConfusion matrix saved to {output_filename}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
    # plt.show() # Uncomment if you want to display the plot interactively

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
        
        true_species_original = entry.get('True Species', '')
        true_species = true_species_original.strip()


        if not path_from_csv or not true_species:
            print(f"Skipping invalid row #{i+1}: Original entry: {entry}")
            results_log.append({
                "filename_csv": path_from_csv_original,
                "true_species": true_species_original,
                "predicted_species": "N/A (Skipped invalid row)",
                "is_correct": False
            })
            continue
        
        print(f"Processing CSV row #{i+1}: '{path_from_csv}' (True species: {true_species})...")
        
        image_full_path = None
        if os.path.isabs(path_from_csv): 
            if os.path.exists(path_from_csv):
                image_full_path = path_from_csv
            else:
                base_name = os.path.basename(path_from_csv)
                image_full_path = find_image_path(base_name, IMAGE_DIRECTORIES)
        else: 
            image_full_path = find_image_path(path_from_csv, IMAGE_DIRECTORIES)

        if not image_full_path or not os.path.exists(image_full_path):
            actual_path_searched = image_full_path if image_full_path else path_from_csv
            print(f"  Image not found or path invalid: {actual_path_searched}")
            results_log.append({
                "filename_csv": path_from_csv_original,
                "true_species": true_species_original,
                "predicted_species": f"N/A (Image not found at '{actual_path_searched}')",
                "is_correct": False
            })
            continue
        
        temp_zoomed_image_path = None
        predicted_species_value = None # Renamed to avoid conflict
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_zoomed_image_path = tmp_file.name
            
            if preprocess_image_and_zoom(image_full_path, temp_zoomed_image_path, ZOOM_CROP_FACTOR):
                # print(f"  Original image: {os.path.basename(image_full_path)}, Zoomed temp image: {os.path.basename(temp_zoomed_image_path)}")
                total_images_processed += 1
                predicted_species_value = run_fishsense_classifier(temp_zoomed_image_path)
            else:
                print(f"  Skipping FishSense classification for {os.path.basename(image_full_path)} due to preprocessing error.")
                results_log.append({
                    "filename_csv": path_from_csv_original,
                    "true_species": true_species_original,
                    "predicted_species": "N/A (Preprocessing failed)",
                    "is_correct": False
                })
                continue 
        finally:
            if temp_zoomed_image_path and os.path.exists(temp_zoomed_image_path):
                # print(f"  Cleaning up temporary file: {temp_zoomed_image_path}")
                os.remove(temp_zoomed_image_path)
        
        current_predicted_species_for_log = "N/A (Prediction failed or None from FishSense)"
        is_correct_flag = False

        if predicted_species_value is not None: 
            current_predicted_species_for_log = predicted_species_value
            # print(f"  Predicted species: {predicted_species_value}")
            is_correct_flag = predicted_species_value.strip().lower() == true_species.strip().lower()
            if is_correct_flag:
                correct_predictions += 1
                # print("  Result: CORRECT")
            # else:
                # print("  Result: INCORRECT")
        # else: # This case is if run_fishsense_classifier returned None after successful preprocessing
            # print(f"  Failed to get prediction for {os.path.basename(image_full_path)} from FishSense.")
            
        results_log.append({
            "filename_csv": path_from_csv_original,
            "true_species": true_species_original, # Store original case for labels if preferred
            "predicted_species": current_predicted_species_for_log,
            "is_correct": is_correct_flag
        })
        print("-" * 30)

    print("\n Evaluation Summary ")
    y_true_for_cm = []
    y_pred_for_cm = []

    for log_entry in results_log:
        status_indicator = "CORRECT" if log_entry["is_correct"] else "INCORRECT"
        details = ""
        true_label = log_entry['true_species'].strip() # Use stripped for CM
        pred_label = log_entry['predicted_species'].strip()

        if "N/A (" not in pred_label: # Only include valid classifications for CM
            y_true_for_cm.append(true_label.lower()) # Use consistent case for CM
            y_pred_for_cm.append(pred_label.lower()) # Use consistent case for CM
        
        if not log_entry["is_correct"]:
            if "Skipped" in pred_label:
                 details = f" (Reason: Skipped invalid CSV row)"
            elif "Image not found" in pred_label:
                details = f" (Reason: Image not found)"
            elif "Preprocessing failed" in pred_label:
                details = f" (Reason: Image preprocessing failed)"
            elif pred_label == "N/A (Prediction failed or None from FishSense)" or pred_label == "None":
                details = f" (Reason: Prediction failed/None from FishSense)"
        
        print(f"File: {log_entry['filename_csv']}, True: {log_entry['true_species']}, Predicted: {log_entry['predicted_species']}, Status: {status_indicator}{details}")

    if total_images_processed > 0:
        accuracy = (correct_predictions / total_images_processed) * 100
        print(f"\nTotal images effectively processed: {total_images_processed} (out of {len(ground_truth_data)} CSV entries)")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")

        all_class_names = sorted(list(set(y_true_for_cm + y_pred_for_cm)))
        if all_class_names:
            plot_confusion_matrix(y_true_for_cm, y_pred_for_cm, all_class_names)
        else:
            print("\nNo valid predictions were made, skipping confusion matrix generation.")
            
    else:
        print(f"No images were successfully processed out of {len(ground_truth_data)} CSV entries.")
        if len(ground_truth_data) > 0:
            print("Please check error messages above for issues like missing files or FishSense output problems.")

if __name__ == "__main__":
    main()