import csv
import subprocess
import os
import re


GROUND_TRUTH_CSV = "/Users/tanishqsingh/Documents/GitHub/fishsense.rs/ground_truth.csv" 
IMAGE_DIRECTORIES = [
    "/Users/tanishqsingh/Downloads/2024.06.27.FishSense.CCFRP/iPhone",
    "/Users/tanishqsingh/Downloads/2024.06.27.FishSense.CCFRP/iPad",
]
FISHSENSE_PROJECT_DIR = os.getcwd() 

def find_image_path(base_filename, directories):
    """Searches for the image (base filename) in the specified directories."""
    for directory in directories:
        path = os.path.join(directory, base_filename)
        if os.path.exists(path):
            return path
    return None

def run_fishsense_classifier(image_path):
    """
    Runs the FishSense classifier on the given image and parses the output.
    Returns the predicted species name or None if not found or on error.
    """
    if not image_path:
        return None

    command = ["cargo", "run", "--release", "--", image_path]
    
    try:
       
        process = subprocess.Popen(command, cwd=FISHSENSE_PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)

        if process.returncode != 0:
            print(f"  Error running FishSense for {os.path.basename(image_path)}:")
            print(f"  Stderr: {stderr.strip()}")
            return None

        match = re.search(r'Classification: Some\("([^"]+)"\)', stdout)
        if match:
            return match.group(1)
        elif "Classification: None" in stdout:
            print(f"  FishSense returned None classification for {os.path.basename(image_path)}")
            return "None" 
        else:
            print(f"  Could not parse species from FishSense output for {os.path.basename(image_path)}:")
            print(f"  Stdout: {stdout.strip()}")
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
        print(f"Please create it with 'File Name,True Species' columns (or check your headers).")
        return

    ground_truth_data = []
    try:
        with open(GROUND_TRUTH_CSV, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                print(f"Error: Could not read field names from '{GROUND_TRUTH_CSV}'. Is it empty or not a valid CSV?")
                return
            print(f"CSV Headers found: {reader.fieldnames}") 
        
            if 'File Name' not in reader.fieldnames or 'True Species' not in reader.fieldnames:
                print("Error: CSV file must contain 'File Name' and 'True Species' columns.")
                print(f"Please ensure your CSV headers match these names exactly (case-sensitive, with space).")
                return

            for row in reader:
                ground_truth_data.append(row)
    except Exception as e:
        print(f"Error reading or parsing CSV file '{GROUND_TRUTH_CSV}': {e}")
        return


    if not ground_truth_data:
        print(f"No data rows found in {GROUND_TRUTH_CSV} after reading headers.")
        return

    correct_predictions = 0
    total_images_processed = 0 
    results_log = [] 

    print(f"\nStarting FishSense classification accuracy evaluation with {len(ground_truth_data)} entries from CSV...\n")

    for i, entry in enumerate(ground_truth_data):
      
        path_from_csv = entry.get('File Name')
        true_species = entry.get('True Species')

        if not path_from_csv or not true_species:
            print(f"Skipping invalid row #{i+1} (missing 'File Name' or 'True Species'): {entry}")
            results_log.append({
                "filename_csv": path_from_csv or "N/A",
                "true_species": true_species or "N/A",
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
               
                print(f"  Absolute path from CSV not found: '{path_from_csv}'. Trying to find base name...")
                base_name = os.path.basename(path_from_csv)
                image_full_path = find_image_path(base_name, IMAGE_DIRECTORIES)
                if image_full_path:
                    print(f"  Found as '{base_name}' at: {image_full_path}")
        else: 
            print(f"  '{path_from_csv}' is a relative path. Searching in IMAGE_DIRECTORIES...")
            image_full_path = find_image_path(path_from_csv, IMAGE_DIRECTORIES)
            if image_full_path:
                 print(f"  Found at: {image_full_path}")


        if not image_full_path or not os.path.exists(image_full_path):
            actual_path_searched = image_full_path if image_full_path else path_from_csv
            print(f"  Image not found or path invalid: {actual_path_searched}")
            results_log.append({
                "filename_csv": path_from_csv,
                "true_species": true_species,
                "predicted_species": f"N/A (Image not found at '{actual_path_searched}')",
                "is_correct": False
            })
            continue
        
        total_images_processed += 1
        predicted_species = run_fishsense_classifier(image_full_path)

        if predicted_species is not None: 
            print(f"  Predicted species: {predicted_species}")
            is_correct = predicted_species.strip().lower() == true_species.strip().lower()
            if is_correct:
                correct_predictions += 1
                print("  Result: CORRECT")
            else:
                print("  Result: INCORRECT")
            results_log.append({
                "filename_csv": path_from_csv,
                "true_species": true_species,
                "predicted_species": predicted_species,
                "is_correct": is_correct
            })
        else:
            print(f"  Failed to get prediction for {os.path.basename(image_full_path)}.")
            results_log.append({
                "filename_csv": path_from_csv,
                "true_species": true_species,
                "predicted_species": "N/A (Prediction failed or None)",
                "is_correct": False
            })
        print("-" * 30)


    print("\nEvaluation Summary")
    for log_entry in results_log:
        status = "CORRECT" if log_entry["is_correct"] else "WRONG"
        reason = ""
        if not log_entry["is_correct"] and "N/A" in log_entry["predicted_species"]:
            reason = f" ({log_entry['predicted_species']})" 

        print(f"{status} File (from CSV): {log_entry['filename_csv']}, True: {log_entry['true_species']}, Predicted: {log_entry['predicted_species']}{reason}")

    if total_images_processed > 0:
        accuracy = (correct_predictions / total_images_processed) * 100
        print(f"\nTotal images effectively processed: {total_images_processed} (out of {len(ground_truth_data)} CSV entries)")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print(f"No images were successfully processed out of {len(ground_truth_data)} CSV entries.")
        if len(ground_truth_data) > 0:
            print("Please check error messages above ")

if __name__ == "__main__":
    main()