from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, logging
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from huggingface_hub import notebook_login
import torch.nn as nn
from tools import *
import torch
import os

huggingface_api_key = "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"
os.environ["HUGGINGFACE_TOKEN"] = huggingface_api_key
notebook_login()
os.environ.pop("HUGGINGFACE_TOKEN", None)

device = "cuda"
checkpoint = "mistralai/Mistral-7B-v0.1"

config = AutoConfig.from_pretrained(checkpoint)
config.update({'sliding_window' : 8_192}) 
config.update({'rope_scaling' : {"type": "yarn",
                                 "factor": 4, 
                                 "original_max_position_embeddings": 8192,
                                 "finetuned": True,
                                }})  

#training_arguments
args_output_dir = "model_weights/Mistral-7B-v0.1-context_extension-stage3"
args_max_steps = 350
args_eval_freq_default = 50
args_log_freq_default = 1
args_save_freq_default = 50
args_batch_size = 1
args_learning_rate = 2e-6
args_lr_scheduler_type = "constant_with_warmup"
args_num_warmup_steps = 50
args_gradient_accumulation_steps_default = 32
args_weight_decay = 0.0
adam_beta1 = 0.9
adam_beta2 = 0.95

#lora fine tuning
lora_r_default = 8
lora_alpha_default = 32
lora_dropout_default = 0.05

logging.set_verbosity_info()


def main():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = False, revision = 'main')
    model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                low_cpu_mem_usage = True,
                                                torch_dtype = torch.float16,
                                                revision = 'main',
                                                device_map = 'auto',
                                                use_flash_attention_2 = True,
                                                config = config,)

    dataset = load_dataset('Yukang/LongAlpaca-12k')['train']
    dataset = dataset.map(lambda x: build_qa_inputs(x, tokenizer))
    dataset = dataset.filter(lambda x: len(x['input_ids']) <= 32_768)
    dataset = dataset.shuffle(seed=42)
    train_dataset, val_dataset = split_dataset(dataset, 0.9)
    datasets = DatasetDict({
        'train': train_dataset,
        'val': val_dataset
    })

    lora_config = LoraConfig(
        r=lora_r_default, 
        lora_alpha=lora_alpha_default, 
        lora_dropout=lora_dropout_default,
        bias="none", 
        task_type="CAUSAL_LM",  
        target_modules = ["q_proj", "k_proj", "v_proj"],
        )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    load_weights(model, './model_weights/Mistral-7B-v0.1-context_extension-stage2/checkpoint_400.pt')
    
    prepare_lora_plus_training(model)

    training_args = TrainingArguments(
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        output_dir=args_output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        #load_best_model_at_end=True,
        dataloader_drop_last=True,
        max_steps=args_max_steps,
        eval_steps=args_eval_freq_default,
        #save_steps=args_save_freq_default,
        logging_steps=args_log_freq_default,
        per_device_train_batch_size=args_batch_size,
        per_device_eval_batch_size=args_batch_size,
        learning_rate=args_learning_rate,
        lr_scheduler_type=args_lr_scheduler_type,
        warmup_steps=args_num_warmup_steps,
        weight_decay=args_weight_decay,
        gradient_accumulation_steps=args_gradient_accumulation_steps_default,
        gradient_checkpointing=True,
        fp16=True,
        push_to_hub=False,
        report_to='wandb',
        run_name=args_output_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        callbacks=[SaveLoraPlusLayersCallback(save_steps=args_save_freq_default, 
                                        layer_names=["lora", "embed_tokens", "norm"], 
                                        output_dir=args_output_dir)],
    )

    trainer.train()


def build_qa_inputs(sample, tokenizer):
    prompt = '[INST] ' + sample['instruction'] + ' [\INST] ' + sample['output']
    inputs = tokenizer(prompt)
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': inputs['input_ids']}


if __name__ == "__main__":
    main()