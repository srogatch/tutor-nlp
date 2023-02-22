import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    # checkpoint = "bigscience/bloomz"
    checkpoint = "EleutherAI/gpt-j-6B"

    tm_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype="auto", device_map="auto", offload_folder="offload")
    tm_end = time.time()
    print(f'Loaded in {tm_end - tm_start} seconds.')

    prompt = "Which investors can I ask for investments in a software reverse architecting project?"

    tm_start = time.time()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    tm_end = time.time()
    print(f'Encoded in {tm_end - tm_start} seconds.')

    tm_start = time.time()
    outputs = model.generate(inputs, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.5)
    tm_end = time.time()
    print(f'Generated in {tm_end - tm_start} seconds.')

    tm_start = time.time()
    response = tokenizer.decode(outputs[0])
    tm_end = time.time()
    print(f'Decoded in {tm_end - tm_start} seconds.')

    print(response)


if __name__ == '__main__':
    main()
