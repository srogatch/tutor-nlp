import os
import time
import readline
import deepspeed

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.distributed import init_process_group, Backend
import torch.multiprocessing as mp

def main(rank: int, size: int):
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    # checkpoint = "bigscience/bloomz"
    checkpoint = "philschmid/gpt-j-6B-fp16-sharded"

    init_process_group(Backend.NCCL, world_size=size, rank=rank, init_method="file:///tmp/llm_nccl1")

    tm_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16)
    tm_end = time.time()
    print(f'Loaded in {tm_end - tm_start} seconds.')

    tm_start = time.time()
    # init deepspeed inference engine
    ds_model = deepspeed.init_inference(
        model=model,  # Transformers models
        mp_size=2,  # Number of GPU
        dtype=torch.float16,  # dtype of the weights (fp16)
        replace_method="auto",  # Lets DS autmatically identify the layer to replace
        replace_with_kernel_inject=True,  # replace the model with the kernel injector
    )
    tm_end = time.time()
    print(f'Deepspeed init in {tm_end - tm_start} seconds.')

    while True:
        prompt = input('Request to LLM: ')

        tm_start = time.time()
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        tm_end = time.time()
        print(f'Encoded in {tm_end - tm_start} seconds.')

        tm_start = time.time()
        outputs = ds_model.generate(inputs, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.5)
        tm_end = time.time()
        print(f'Generated in {tm_end - tm_start} seconds.')

        tm_start = time.time()
        response = tokenizer.decode(outputs[0])
        tm_end = time.time()
        print(f'Decoded in {tm_end - tm_start} seconds.')

        print(response)


if __name__ == '__main__':
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=main, args=(rank, size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
