from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt

device = "cuda"
context_lengths = [500, 2000, 4000, 8000, 16000, 24000, 32000]
checkpoint = "mistralai/Mistral-7B-v0.1"
#checkpoint_lora = "sade-adrien/Mistral-7B-Instruct-v0.1-LC16k-PI-v2"
#checkpoint_lora = "sade-adrien/Mistral-7B-Instruct-v0.1-LC16k-PI"
#checkpoint = "lmsys/vicuna-13b-v1.3"
#checkpoint = "NousResearch/Yarn-Mistral-7b-128k"



config = AutoConfig.from_pretrained(checkpoint)
config.update({'sliding_window' : 40_000})  #eliminating sliding window
config.update({'rope_scaling' : {"type": "yarn",
                                 "factor": 2,
                                 "original_max_position_embeddings": 8192,
                                 "finetuned": False,
                                }})  


def main():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = False, revision = 'main')
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                        use_flash_attention_2=True,
                                        torch_dtype=torch.bfloat16,
                                        device_map="auto",
                                        config = config,)
                                        
    dataset = load_dataset('sade-adrien/redpajama_v2_sample_1M')['train']

    PPL = []
    for context_len in context_lengths:
        dataset_select = dataset.filter(lambda x: len(x['raw_content']) >= context_len * 6)
        dataset_select = dataset_select.select(torch.arange(400))

        ppl = run_test(model, tokenizer, dataset_select, context_len)
        PPL.append(ppl)
        print(f"For context length {context_len}, Avg Perplexity = {ppl}\n" )

    for context_len, ppl in zip(context_lengths, PPL):
        print(f"For context length {context_len}, Avg Perplexity = {ppl}" )


def run_test(model, tokenizer, dataset, context_length):
    model.eval()
    total_loss = 0
    PPL = []

    with torch.no_grad():
        i = 0
        for data in tqdm(dataset):
            if i==0:
                ...
                #input(data['raw_content'])
            inputs = tokenizer(data['raw_content'], truncation=True, max_length=context_length, return_tensors='pt').to(device)

            loss = model(**inputs, labels=inputs['input_ids']).loss
            total_loss += loss
            PPL.append(torch.exp(total_loss/(i+1)).item())

            i += 1
    

    plt.plot(PPL)
    plt.savefig(f'ppl_{context_length}.png')
    plt.show()
    plt.clf()
    
    model.train()
    return PPL[-1]



if __name__ == '__main__':
    main()
    