import os
import json
from collections import defaultdict
import shutil

NUM_DEL = 5

# Function to parse JSON configuration files
def parse_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to delete model-related files in all folders
def delete_model_files(model_name):
    folders = ['reconstruction_plots', 'weights', 'configs', 'runs']
    for folder in folders:
        for filename in os.listdir(folder):
            if model_name in filename:
                file_path = os.path.join(folder, filename)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

                print(f"Deleted: {file_path}")

# Set base directory
config_dir = 'configs'

# Step 1: Collect valid configs
models_by_dataset = defaultdict(list)
valid_model_names = set()

for file in os.listdir(config_dir):
    if file.startswith('run_') and file.endswith('.json'):
        file_path = os.path.join(config_dir, file)
        try:
            config = parse_config(file_path)
            model_name = os.path.splitext(file)[0]
            dataset = config.get('dataset')
            if dataset is not None:
                models_by_dataset[dataset].append((model_name, config))
                valid_model_names.add(model_name)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

# Step 2: Keep top 2 models per dataset, delete the rest
for dataset, models in models_by_dataset.items():
    models_sorted = sorted(models, key=lambda x: x[1].get('test_loss', float('inf')))
    models_to_keep = set(model_name for model_name, _ in models_sorted[:NUM_DEL])
    models_to_delete = set(model_name for model_name, _ in models_sorted[NUM_DEL:])

    for model_name in models_to_delete:
        delete_model_files(model_name)

# Step 3: Remove orphaned files (no config)
folders_to_check = ['reconstruction_plots', 'weights', 'runs']

for folder_path in folders_to_check:
    for filename in os.listdir(folder_path):
        for valid_model in valid_model_names:
            if valid_model in filename:
                break
        else:
            # No matching config -> delete the file
            file_path = os.path.join(folder_path, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            print(f"Deleted orphaned file: {file_path}")
