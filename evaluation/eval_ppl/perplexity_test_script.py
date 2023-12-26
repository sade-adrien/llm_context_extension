from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import torch

sys.path.append('/mnt/datascience1/Adrien/context_extension')
from tools import *


device = "cuda"
context_lengths = [500, 2000, 4000, 8000, 16000, 24000, 32000]
checkpoint = "mistralai/Mistral-7B-v0.1"

config = AutoConfig.from_pretrained(checkpoint)
config.update({'sliding_window' : 8_192}) 
config.update({'rope_scaling' : {"type": "yarn",
                                 "factor": 4, 
                                 "original_max_position_embeddings": 8192,
                                 "finetuned": True,
                                }})  

lora_r_default = 8
lora_alpha_default = 32
lora_dropout_default = 0.05
lora_config = LoraConfig(
        r=lora_r_default, 
        lora_alpha=lora_alpha_default, 
        lora_dropout=lora_dropout_default,
        bias="none", 
        task_type="CAUSAL_LM",  
        target_modules = ["q_proj", "k_proj", "v_proj"],
        )


def main():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = False, revision = 'main')
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                low_cpu_mem_usage = True,
                                                torch_dtype = torch.float16,
                                                revision = 'main',
                                                device_map = 'auto',
                                                use_flash_attention_2 = True,
                                                config = config,)
    

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    load_weights(model, '../../model_weights/Mistral-7B-v0.1-context_extension-stage2/checkpoint_400.pt')
                                        
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
    

    #plt.plot(PPL)
    #plt.savefig(f'ppl_{context_length}.png')
    #plt.show()
    #plt.clf()
    
    model.train()
    return PPL[-1]



if __name__ == '__main__':
    main()
    