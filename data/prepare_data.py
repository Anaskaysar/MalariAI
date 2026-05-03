import pandas as pd
from tqdm import tqdm
import argparse
import os

def parse_json_to_csv(json_path: str, output_csv_path: str, image_dir_prefix: str):
    """
    Reads a bounding box JSON annotation file and converts it into a CSV format.
    Automatically handles normalization for MalariAI dataset categories.
    """
    print(f"Reading {json_path}...")
    try:
        data_json = pd.read_json(json_path)
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}")
        return

    data = []
    # Using the structure from Previous/preprocessing.py
    for i in tqdm(range(data_json.shape[0]), desc=f"Processing {os.path.basename(json_path)}"):
        # The structure handles typical object nested fields
        objects_list = data_json.iloc[i, 1]
        
        # Check if objects_list is iterable
        if not isinstance(objects_list, list):
            continue
            
        for j in range(len(objects_list)):
            try:
                # The pathname structure is assumed to be something like 'images/train/image_name.jpg'
                img_name_parts = data_json.iloc[i, 0]['pathname'].split('/')
                img_name = img_name_parts[-1] if len(img_name_parts) > 0 else "unknown.jpg"
                
                label = objects_list[j]['category']
                
                # Bounding boxes
                x_min = objects_list[j]['bounding_box']['minimum']['c']
                x_max = objects_list[j]['bounding_box']['maximum']['c']
                y_min = objects_list[j]['bounding_box']['minimum']['r']
                y_max = objects_list[j]['bounding_box']['maximum']['r']
                
                data.append([img_name, label, x_min, y_min, x_max, y_max])
            except (KeyError, TypeError) as e:
                # In case a row is missing bounding box info or doesn't match format
                continue
                
    df = pd.DataFrame(data, columns=['img_name', 'label', 'x_min', 'y_min', 'x_max', 'y_max'])
    
    # Prefix image names with the correct directory path as used in training
    df['img_name'] = df['img_name'].apply(lambda x: f"{image_dir_prefix}/{x}")
    
    # Preprocessing step 1: Remove 'difficult' category if present
    df = df[df['label'] != "difficult"]
    
    # Preprocessing step 2: Convert specific detailed labels to a general 'non_rbc' class
    non_rbc = ['trophozoite', 'schizont', 'ring', 'gametocyte', 'leukocyte']
    df['label'] = df['label'].apply(lambda x: 'non_rbc' if x in non_rbc else x)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved preprocessed annotations to {output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Compile MalariAI JSON annotations into processed CSV files.")
    parser.add_argument("--train-json", type=str, default="input/training.json", help="Path to training JSON file")
    parser.add_argument("--test-json", type=str, default="input/test.json", help="Path to test JSON file")
    parser.add_argument("--train-csv", type=str, default="data/processed/train_annotation.csv", help="Output path for processed train CSV")
    parser.add_argument("--test-csv", type=str, default="data/processed/test_annotation.csv", help="Output path for processed test CSV")
    args = parser.parse_args()

    parse_json_to_csv(args.train_json, args.train_csv, "training_images")
    parse_json_to_csv(args.test_json, args.test_csv, "testing_images")

if __name__ == "__main__":
    main()
