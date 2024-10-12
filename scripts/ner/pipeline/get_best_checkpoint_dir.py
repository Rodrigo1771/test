import json
import argparse
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

# Get best checkpoint info
checkpoints = os.listdir(args.output_dir)
best_checkpoint_info = {'f1': -1, 'path': ''}
for checkpoint in checkpoints:
    checkpoint_trainer_state_path = os.path.join(args.output_dir, checkpoint, "trainer_state.json")
    with open(checkpoint_trainer_state_path, 'r') as f:
        state = json.load(f)
    f1, path = state["best_metric"], state["best_model_checkpoint"]
    if f1 > best_checkpoint_info['f1']:
        best_checkpoint_info['path'] = path

# Print the dir path of the best checkpoint
print(best_checkpoint_info['path'])
