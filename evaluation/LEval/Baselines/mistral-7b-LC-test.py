import math
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
# -*- coding:utf-8 -*-
import argparse
from LEval_config import *
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
sys.path.append('/mnt/datascience1/Adrien/context_extension')
from tools import *


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        B_INST, E_INST = "[INST]", "[\INST]" ################ to switch back to /
        sys_prompt = get_sys_prompt(args, file_name)

        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, tokenizer) > max_length:
                document = " ".join(document.split(" ")[:max_length - cnt])  # chunk the input len into 16k tokens
                cnt += 250

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "code" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + sys_prompt + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + sys_prompt + context + E_INST
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst}.  Please directly give answer without any additonal output or explanation "
                    message = B_INST + sys_prompt + context + E_INST
                    message += "\nAnswer:"
                else:
                    context = "Document is as follows. {document} Instruction: {inst} " + f"\nAnswer this question with {len(out.split())} words."
                    message = B_INST + sys_prompt + context + E_INST
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long document>")

                inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                sample = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens)
                prompt_length = inputs.input_ids.size()[-1]
                output = tokenizer.decode(sample[0][prompt_length:])

                save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
                save_d['evaluation'] = d['evaluation']

                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "\nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    sample = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens)
                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[0][prompt_length:])
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("[document]:",text_inputs[:100] + "...")
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
                # break
        fw.close()
        # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', default="4k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=int, default=0)

    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')

    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')
    parser.add_argument('--flash', action='store_true', help='set this if you want to use flash attention')
    args = parser.parse_args()

    model_path = "mistralai/Mistral-7B-v0.1"
    open_source_model = "mistralai/Mistral-7B-v0.1-context_extension-stage3-" + args.max_length #+ "-noSW"

    max_length = k_to_number(args.max_length) - max_new_tokens

    config = AutoConfig.from_pretrained(model_path)
    config.update({'sliding_window' : 8_192}) 
    config.update({'rope_scaling' : {"type": "yarn",
                                    "factor": 4, 
                                    "original_max_position_embeddings": 8192,
                                    "finetuned": True,
                                    }})  
    if args.flash:
        config.update({'_flash_attn_2_enabled' : True})  #Flash Attention

    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = False, revision = 'main')
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                low_cpu_mem_usage = True,
                                                torch_dtype = torch.bfloat16,
                                                revision = 'main',
                                                device_map = 'auto',
                                                config = config,)
    
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none", 
        task_type="CAUSAL_LM",  
        target_modules = ["q_proj", "k_proj", "v_proj"],
        )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    load_weights(model, '../../model_weights/Mistral-7B-v0.1-context_extension-stage3/checkpoint_350.pt')

    model = model.eval()

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
