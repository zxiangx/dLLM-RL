import json

input_file = 'train_math.jsonl'
output_file = 'sorted_reverse_train_math.jsonl'

data = []

# Read the input file
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Sort the data based on the length of the 'prompt' field
data.sort(key=lambda x: -len(x['prompt']))

# Write the sorted data to the new file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Sorted data written to {output_file}")