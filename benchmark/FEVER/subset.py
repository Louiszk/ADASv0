# https://fever.ai/dataset/fever.html
import json
import random
from collections import defaultdict

def extract_fever_subset(input_file, output_file, samples_per_label=40, seed=42):
    """
    Extract a random subset from the FEVER dataset with equal representation of each label.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSON file
        samples_per_label: Number of samples to extract for each label
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the input file and group by label
    samples_by_label = defaultdict(list)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            label = sample.get('label')
            if label:
                del sample["evidence"]
                samples_by_label[label].append(sample)
    
    # Select random samples for each label
    selected_samples = []
    for label, samples in samples_by_label.items():
        if len(samples) >= samples_per_label:
            selected = random.sample(samples, samples_per_label)
            selected_samples.extend(selected)
            print(f"Selected {len(selected)} samples for label '{label}'")
        else:
            selected_samples.extend(samples)
            print(f"Warning: Only {len(samples)} samples available for label '{label}'")
    
    # Write the selected samples to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_samples, f, indent=2)
    
    print(f"Extracted {len(selected_samples)} samples to {output_file}")
    
    # Return statistics
    return {label: len([s for s in selected_samples if s.get('label') == label]) 
            for label in samples_by_label.keys()}

if __name__ == "__main__":
    input_file = "benchmark/Fever/train.jsonl"
    output_file = "benchmark/Fever/fever_subset.json"
    stats = extract_fever_subset(input_file, output_file)
    print("Statistics of the extracted subset:")
    for label, count in stats.items():
        print(f"  {label}: {count}")