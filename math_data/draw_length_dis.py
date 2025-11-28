import json
import os

import matplotlib.pyplot as plt

def draw_length_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    lengths = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "prompt" in data:
                        lengths.append(len(data["prompt"]))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not lengths:
        print("No valid data found to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Prompt Lengths')
    plt.xlabel('Length of Prompt')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    
    output_file = 'prompt_length_distribution.png'
    plt.savefig(output_file)
    print(f"Histogram saved to {output_file}")
    # plt.show() # Uncomment if running in an environment with display support

if __name__ == "__main__":
    file_path = 'train_math.jsonl'
    draw_length_distribution(file_path)