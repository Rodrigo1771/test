import json
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--language")
parser.add_argument("--model_id")
args = parser.parse_args()

# Load results json
hyperparameter_search_dir = f'out/{args.dataset}/hyperparameter-search/{args.language}'
with open(f"{hyperparameter_search_dir}/{args.model_id}_training_results.json", 'r') as f:
    results = json.load(f)

# Find the best overall model
best_overall_model = {'accuracy': 0}
for experiment in results.items():
    hyperparameters, properties = experiment
    batch_size, learning_rate = hyperparameters.split('_')
    curr_acc = properties['best_model']['accuracy']
    if curr_acc > best_overall_model['accuracy']:
        best_overall_model = {
            'batch_size': batch_size,
            'learning_rate': f'{learning_rate[:-2]}{learning_rate[-1]}',
            'accuracy': curr_acc,
            'epoch': properties['best_model']['epoch']
        }

# Log best overall model info to results file
results['best_overall_model'] = best_overall_model
with open(f"{hyperparameter_search_dir}/{args.model_id}_training_results.json", 'w') as file:
    json.dump(results, file, indent=4)
