import os
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder

normal_datasets_and_info = {
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
    },
    'combined-train-distemist-dev': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/combined-train-distemist-dev_loading_script.py'
    },
    'combined-train-drugtemist-es-dev': {
        'conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/combined-train-drugtemist-es-dev_loading_script.py'
    }
}

augmented_datasets_and_info = {
    'symptemist': {
        'augmented_train_conll_file_path': '../../data-aug/augment/ner/out/<MODEL_TYPE>/symptemist/train_aug_3_<DISTANCE_THRESHOLD>.conll',
        'dev_and_test_conll_files_dir': 'symptemist-parse/out/',
        'loading_script': 'symptemist-parse/symptemist_loading_script.py'
    },
    'cantemist': {
        'augmented_train_conll_file_path': '../../data-aug/augment/ner/out/<MODEL_TYPE>/cantemist/train_aug_3_<DISTANCE_THRESHOLD>.conll',
        'dev_and_test_conll_files_dir': 'cantemist-parse/out/',
        'loading_script': 'cantemist-parse/cantemist_loading_script.py'
    },
    'distemist': {
        'augmented_train_conll_file_path': '../../data-aug/augment/ner/out/<MODEL_TYPE>/multicardioner/distemist_train_aug_3_<DISTANCE_THRESHOLD>.conll',
        'dev_and_test_conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/distemist_loading_script.py'
    },
    'drugtemist-es': {
        'augmented_train_conll_file_path': '../../data-aug/augment/ner/out/<MODEL_TYPE>/multicardioner/drugtemist_es_train_aug_3_<DISTANCE_THRESHOLD>.conll',
        'dev_and_test_conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/drugtemist-es_loading_script.py'
    },
    'drugtemist-en': {
        'augmented_train_conll_file_path': '../../data-aug/augment/ner/out/<MODEL_TYPE>/multicardioner/drugtemist_en_train_aug_3_<DISTANCE_THRESHOLD>.conll',
        'dev_and_test_conll_files_dir': 'multicardioner-parse/out/',
        'loading_script': 'multicardioner-parse/loading-scripts/drugtemist-en_loading_script.py'
    },
    'drugtemist-it': {
        'augmented_train_conll_file_path': '../../data-aug/augment/ner/out/<MODEL_TYPE>/multicardioner/drugtemist_it_train_aug_3_<DISTANCE_THRESHOLD>.conll',
        'dev_and_test_conll_files_dir': 'multicardioner-parse/out/',
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

# Delete repo's if they already exist
for dataset, _ in tqdm(normal_datasets_and_info.items()):
    hf_repo_id = f'{hf_username}/{dataset}-ner'
    api.delete_repo(repo_id=hf_repo_id, missing_ok=True, repo_type='dataset', token=hf_token)

for model_type in ['word2vec', 'fasttext']:
    for distance_threshold in [75, 8, 85, 9]:
        for dataset, _ in tqdm(augmented_datasets_and_info.items()):
            hf_repo_id = f'{hf_username}/{dataset}-{model_type}-{distance_threshold}-ner'
            api.delete_repo(repo_id=hf_repo_id, missing_ok=True, repo_type='dataset', token=hf_token)

# Upload normal datasets
for dataset, info in tqdm(normal_datasets_and_info.items()):
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
    data = data.replace('<MODEL_TYPE_AND_DISTANCE_THRESHOLD>', '')
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
    data = data.replace('ner/resolve/main/', '<MODEL_TYPE_AND_DISTANCE_THRESHOLD>ner/resolve/main/')
    with open(os.path.join(info["loading_script"]), 'w') as f:
        f.write(data)

    # Upload splits
    if 'combined' in dataset:
        api.upload_file(
            path_or_fileobj=f'{info["conll_files_dir"]}/combined_train.conll',
            path_in_repo='train.conll',
            repo_id=hf_repo_id,
            repo_type='dataset',
            token=hf_token
        )
        for split in ['dev', 'test']:
            if 'distemist' in dataset:
                local_path = f'{info["conll_files_dir"]}/distemist_{split}.conll'
            else:
                local_path = f'{info["conll_files_dir"]}/drugtemist_es_{split}.conll'
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f'{split}.conll',
                repo_id=hf_repo_id,
                repo_type='dataset',
                token=hf_token
            )
    else:
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

# Upload augmented datasets
for model_type in ['word2vec', 'fasttext']:
    for distance_threshold in [75, 8, 85, 9]:
        for dataset, info in tqdm(augmented_datasets_and_info.items()):
            # Create repo
            hf_repo_id = f'{hf_username}/{dataset}-{model_type}-{distance_threshold}-ner'
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
            data = data.replace('<MODEL_TYPE_AND_DISTANCE_THRESHOLD>', f'{model_type}-{distance_threshold}-')
            with open(os.path.join(info["loading_script"]), 'w') as f:
                f.write(data)
            api.upload_file(
                path_or_fileobj=f'{info["loading_script"]}',
                path_in_repo=f'{dataset}-{model_type}-{distance_threshold}-ner.py',
                repo_id=hf_repo_id,
                repo_type='dataset',
                token=hf_token
            )
            with open(os.path.join(info["loading_script"]), 'r') as f:
                data = f.read()
            data = data.replace(hf_username, '<HF_USERNAME>')
            data = data.replace(f'{model_type}-{distance_threshold}-', '<MODEL_TYPE_AND_DISTANCE_THRESHOLD>')
            with open(os.path.join(info["loading_script"]), 'w') as f:
                f.write(data)

            # Upload splits
            train_file_path = (
                info["augmented_train_conll_file_path"]
                .replace('<MODEL_TYPE>', model_type)
                .replace('<DISTANCE_THRESHOLD>', str(distance_threshold/100) if distance_threshold > 10 else str(distance_threshold/10))
            )
            api.upload_file(
                path_or_fileobj=train_file_path,
                path_in_repo=f'train.conll',
                repo_id=hf_repo_id,
                repo_type='dataset',
                token=hf_token
            )
            for split in ['dev', 'test']:
                if 'multicardioner' not in info["dev_and_test_conll_files_dir"]:
                    local_path = f'{info["dev_and_test_conll_files_dir"]}/{split}.conll'
                else:
                    local_path = f'{info["dev_and_test_conll_files_dir"]}/{dataset.replace("-", "_")}_{split}.conll'
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=f'{split}.conll',
                    repo_id=hf_repo_id,
                    repo_type='dataset',
                    token=hf_token
                )

# Delete local .gitattributes
os.remove('.gitattributes')
