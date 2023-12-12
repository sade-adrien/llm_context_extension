---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.1
tags:
- generated_from_trainer
model-index:
- name: Mistral-7B-Instruct-v0.1-LC16k-PI
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Mistral-7B-Instruct-v0.1-LC16k-PI

This model is a fine-tuned version of [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6455

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 32
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 30
- training_steps: 1000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 1.6742        | 0.12  | 100  | 1.6880          |
| 1.6683        | 0.24  | 200  | 1.6711          |
| 1.7301        | 0.36  | 300  | 1.6636          |
| 1.6867        | 0.47  | 400  | 1.6588          |
| 1.4718        | 0.59  | 500  | 1.6557          |
| 1.6843        | 0.71  | 600  | 1.6519          |
| 1.5966        | 0.83  | 700  | 1.6492          |
| 1.9016        | 0.95  | 800  | 1.6472          |
| 1.7488        | 1.07  | 900  | 1.6461          |
| 1.5596        | 1.19  | 1000 | 1.6455          |


### Framework versions

- Transformers 4.35.2
- Pytorch 2.0.0+cu117
- Datasets 2.14.6
- Tokenizers 0.14.1
