import argparse
import nltk
from konlpy.tag import Okt
import os
import json

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def main(args):
    # Ensure necessary NLTK data files are available
    nltk.download('punkt')

    # Load stopwords from the file and remove duplicates
    stop_words = None
    with open("stop_words.txt", 'r', encoding='utf-8') as file:
        stop_words = set(file.read().split())

    # Load the input JSON file
    if not os.path.exists(args.input_json):
        print(f"File {args.input_json} not found.")
        return
    
    data = load_json(args.input_json)

    okt = Okt()
    text = " ".join(utterance['utterance'] for conversation in data for utterance in conversation['input']['conversation'])

    # Calculate global word frequencies
    word_frequency = {}
    for noun in okt.nouns(text):
        word_frequency[noun] = word_frequency.get(noun, 0) + 1

    global_rank = {word: rank for rank, (word, _) in enumerate(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True))}
    global_len = len(global_rank)

    # Process each conversation and calculate local word frequencies
    for conversation in data:
        text = " ".join(utterance['utterance'] for utterance in conversation['input']['conversation'])

        local_word_frequency = {}
        for noun in okt.nouns(text):
            local_word_frequency[noun] = local_word_frequency.get(noun, 0) + 1

        scored_keywords = []
        for word, freq in sorted(local_word_frequency.items(), key=lambda x: x[1], reverse=True):
            if word in stop_words:
                continue
            global_word_rank = global_rank.get(word, global_len)
            score = round(((global_word_rank + 1) / global_len) / ((local_word_frequency[word] + 1) / freq), 2)
            scored_keywords.append((word, score))

        scored_keywords = sorted(scored_keywords, key=lambda x: x[1], reverse=True)[:int(args.top_n)]

        # Update the 'subject_keyword' field in the JSON data
        conversation['input'].setdefault('subject_keyword', []).extend(word for word, _ in scored_keywords)
        conversation['input']['subject_keyword'] = list(set(conversation['input']['subject_keyword']))

    # Save the updated JSON data to the output file
    with open(args.output_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent='\t', ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="add subject",
        description="Testing about Conversational Context Inference."
    )
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON filename")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON filename")
    parser.add_argument("--top_n", type=str, required=True, help="top N keywords")
    
    args = parser.parse_args()
    main(args)
