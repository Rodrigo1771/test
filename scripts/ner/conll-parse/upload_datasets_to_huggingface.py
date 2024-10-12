import os
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder

datasets_and_info = {
    'symptemist': {
        'conll_files_dir': 'symptemist-parse/out/',
        'loading_script': 'symptemist-parse/symptemist_loading_script.py'
    },
    'cantemist': {
        'conll_files_dir': 'cantemist-parse/out/',
        'loading_script': 'cantemist-parse/cantemist_loading_script.py'
    },
    'distemist': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/distemist_loading_script.py'
    },
    'drugtemist': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/drugtemist_loading_script.py'
    },
    'drugtemist-en': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/drugtemist-en_loading_script.py'
    },
    'drugtemist-it': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/drugtemist-it_loading_script.py'
    }
}

# Read hf username and personal token
config = {}
with open('../../config', 'r') as f:
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

# Create local .gitattributes
with open(".gitattributes", "w") as temp_file:
    temp_file.write("*.conll filter=lfs diff=lfs merge=lfs -text")

# Upload datasets
for dataset, info in tqdm(datasets_and_info.items()):
    hf_repo_id = f'{hf_username}/{dataset}-ner'
    api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type='dataset', token=hf_token)
    api.upload_file(
        path_or_fileobj='.gitattributes',
        path_in_repo='.gitattributes',
        repo_id=hf_repo_id,
        repo_type='dataset',
        token=hf_token
    )
    api.upload_file(
        path_or_fileobj=f'{info["loading_script"]}',
        path_in_repo=f'{dataset}-ner.py',
        repo_id=hf_repo_id,
        repo_type='dataset',
        token=hf_token
    )
    for split in ['train', 'dev', 'test']:
        if 'multicardioner' not in info["conll_files_dir"]:
            local_path = f'{info["conll_files_dir"]}/{split}.conll'
        else:
            local_path = f'{info["conll_files_dir"]}/{dataset.replace("-", "_")}_{split}.conll'
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f'{split}.conll',
            repo_id=hf_repo_id,
            repo_type='dataset',
            token=hf_token
        )

# Delete local .gitattributes
os.remove('.gitattributes')
