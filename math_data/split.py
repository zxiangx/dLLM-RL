import json
import random
import os

def split_math_data():
    input_file = 'MATH_train.json'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle the data to ensure random distribution
    random.seed(42)  # Set seed for reproducibility
    random.shuffle(data)

    # Define split sizes
    train_size = 6400
    test_size = 960
    val_size = 320
    
    total_needed = train_size + test_size + val_size
    
    if len(data) < total_needed:
        print(f"Warning: Not enough data. Have {len(data)}, need {total_needed}.")
        # Adjust logic if you want to handle smaller datasets, 
        # but here we proceed assuming the user knows the file size or accepts truncation/errors.
        # For safety, let's just slice what we can or error out. 
        # Proceeding with slicing based on available data.

    # Slice the data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:train_size + test_size + val_size]

    # Helper function to write JSONL
    def write_jsonl(filename, dataset):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                new_item = {
                    "prompt": item.get("question", "")+" You need to put your final answer in \\boxed{}.",
                    "answer": item.get("ground_truth_answer", "")
                }
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        print(f"Created {filename} with {len(dataset)} records.")

    # Write the output files
    write_jsonl('train_math.jsonl', train_data)
    write_jsonl('test_math.jsonl', test_data)
    write_jsonl('val_math.jsonl', val_data)

if __name__ == "__main__":
    split_math_data()