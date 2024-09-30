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
    with open("stop_words.txt", 'r', encoding='utf-8') as file:
        stop_words = file.read().split()

    stop_words = list(set(stop_words))

    with open("stop_words.txt", 'w', encoding='utf-8') as file:
        file.write(" ".join(stop_words))

    # Load the input JSON file
    if not os.path.exists(args.input_json):
        print(f"File {args.input_json} not found.")
        return
    
    data = load_json(args.input_json)

    okt = Okt()
    text = " ".join([utterance['utterance'] for conversation in data for utterance in conversation['input']['conversation']])

    # Calculate global word frequencies
    word_frequency = {}
    for noun in okt.nouns(text):
        word_frequency[noun] = word_frequency.get(noun, 0) + 1

    sorted_global_freq = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
    global_rank = {word: rank for rank, (word, _) in enumerate(sorted_global_freq)}

    global_len = len(global_rank)

    # Process each conversation and calculate local word frequencies
    for conversation in data:
        text = " ".join([utterance['utterance'] for utterance in conversation['input']['conversation']])

        local_word_frequency = {}
        for noun in okt.nouns(text):
            local_word_frequency[noun] = local_word_frequency.get(noun, 0) + 1

        sorted_local_freq = sorted(local_word_frequency.items(), key=lambda x: x[1], reverse=True)

        # Calculate the significance of local keywords
        scored_keywords = []
        for rank, (word, freq) in enumerate(sorted_local_freq):
            if word in stop_words:
                continue
            global_word_rank = global_rank.get(word, global_len)
            score = round(((global_word_rank + 1) / global_len) / ((rank + 1) / len(sorted_local_freq)), 2)
            scored_keywords.append((word, score))

        scored_keywords = sorted(scored_keywords, key=lambda x: x[1], reverse=True)

        # Select top keywords based on the computed score
        top_keywords = [word for word, _ in scored_keywords[:int(args.top_n)]]

        # Update the 'subject_keyword' field in the JSON data
        conversation['input'].setdefault('subject_keyword', []).extend(top_keywords)
        conversation['input']['subject_keyword'] = list(set(conversation['input']['subject_keyword']))

    # Save the updated JSON data to the output file
    with open(args.output_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

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
