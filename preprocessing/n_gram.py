import json
from collections import defaultdict

def generate_ngrams(word, n):
    return [word[i:i+n] for i in range(len(word) - n + 1)]

def build_ngram_index(file_path, n):
    ngram_index = defaultdict(set)

    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip()
            
            if len(word) < 2:
                continue
            
            ngrams = generate_ngrams(word, n)
            
            for ngram in ngrams:
                ngram_index[ngram].add(word)

    ngram_index = {key: list(value) for key, value in ngram_index.items()}
    
    with open("index.json", "w") as index_file:
        json.dump(ngram_index, index_file)

build_ngram_index("dict.txt", n=2)
