import argparse
import json
import tqdm
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def remove_duplicate_sentences(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    
    # Join the sentences back into a single string
    return ' '.join(unique_sentences)

# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
# fmt: on


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # Set eos_token_id properly
    eos_token_id = tokenizer.eos_token_id

    while True:
        user_input = []
        while True:
            input_str = input()
            if input_str.lower() == 'exit':
                break
            else:
                user_input.append(input_str)

        inputs = tokenizer("\n".join(user_input), return_tensors="pt").to(args.device)
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=1024,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        generated_text = (generated_text)
        print("Generated Output: ", generated_text)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
