import json, os, glob

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data: f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_jsonl_files(directory):
    return sorted(glob.glob(os.path.join(directory, "*.jsonl")))