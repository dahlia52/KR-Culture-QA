from datasets import load_dataset
import os

dataset = load_dataset("Chang-Su/ko-wiki-250611", split='train')

current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, 'kowiki_dataset')
dataset.save_to_disk(save_path)

print(f"Saved dataset to '{save_path}' successfully.")