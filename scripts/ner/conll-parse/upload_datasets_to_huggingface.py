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
    'drugtemist-es': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/drugtemist-es_loading_script.py'
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

for dataset, _ in tqdm(datasets_and_info.items()):
    hf_repo_id = f'{hf_username}/{dataset}-ner'
    api.delete_repo(repo_id=hf_repo_id, missing_ok=True, repo_type='dataset', token=hf_token)

# Upload datasets
for dataset, info in tqdm(datasets_and_info.items()):
    # Create repo
    hf_repo_id = f'{hf_username}/{dataset}-ner'
    api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type='dataset', token=hf_token)

    # Upload .gitattributes
    api.upload_file(
        path_or_fileobj='.gitattributes',
        path_in_repo='.gitattributes',
        repo_id=hf_repo_id,
        repo_type='dataset',
        token=hf_token
    )

    # Update and upload loading script
    with open(os.path.join(info["loading_script"]), 'r') as f:
        data = f.read()
    data = data.replace('<HF_USERNAME>', hf_username)
    with open(os.path.join(info["loading_script"]), 'w') as f:
        f.write(data)
    api.upload_file(
        path_or_fileobj=f'{info["loading_script"]}',
        path_in_repo=f'{dataset}-ner.py',
        repo_id=hf_repo_id,
        repo_type='dataset',
        token=hf_token
    )
    with open(os.path.join(info["loading_script"]), 'r') as f:
        data = f.read()
    data = data.replace(hf_username, '<HF_USERNAME>')
    with open(os.path.join(info["loading_script"]), 'w') as f:
        f.write(data)

    # Upload splits
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
