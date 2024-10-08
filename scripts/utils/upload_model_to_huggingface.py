import os
import sys
import logging
import argparse
from huggingface_hub import HfApi, HfFolder

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--local_model_dir", type=str, required=True)
parser.add_argument("--augmented_dataset", type=str)
args = parser.parse_args()

# Logger
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%I:%M:%S %p')
console = logging.StreamHandler(sys.stdout)
console.setFormatter(fmt)
LOGGER.addHandler(console)

# Build hf repo id
path_parts = args.local_model_dir.split('/')
model_name_parts = path_parts[-1].split('_')
model_name_parts = model_name_parts[:-4] if args.augmented_dataset else model_name_parts[:-3]
model_name = '_'.join(model_name_parts) if len(model_name_parts) > 1 else model_name_parts[0]
language = path_parts[-2]
dataset = path_parts[-5] if args.augmented_dataset else path_parts[-4]
aug = f"{args.augmented_dataset}-{path_parts[-1].split('_')[-1]}-" if args.augmented_dataset else ''
hf_repo_id = f"{model_name}-{dataset}-{language}-{aug}el"

# Read hf username and personal token
config = {}
with open('../config', 'r') as f:
    for line in f:
        key, value = line.strip().split('=')
        config[key] = value
    hf_username, hf_token = config.get('hf_username'), config.get('hf_token')
    if hf_username == "<HF_USERNAME>" or hf_token == "<HF_TOKEN>":
        raise ValueError("Invalid credentials: hf_username and/or hf_token are placeholders.")

# Set the token for authentication
HfFolder.save_token(hf_token)

# Initialize the Hugging Face API
api = HfApi()

# Create the repository if it doesn't exist
api.create_repo(hf_repo_id, exist_ok=True, token=hf_token)

# Delete all existing files in the repository
repo_files = api.list_repo_files(repo_id=f'{hf_username}/{hf_repo_id}')
for file in repo_files:
    if file != '.gitattributes':
        api.delete_file(path_in_repo=file, repo_id=f'{hf_username}/{hf_repo_id}', token=hf_token)

# List all files in the local directory and upload them
for filename in os.listdir(args.local_model_dir):
    file_path = os.path.join(args.local_model_dir, filename)
    relative_path = os.path.relpath(str(file_path), args.local_model_dir)

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=relative_path,
        repo_id=f'{hf_username}/{hf_repo_id}',
        token=hf_token
    )

LOGGER.info(f"Model uploaded successfully to {hf_username}/{hf_repo_id}")
