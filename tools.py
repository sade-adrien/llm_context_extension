from transformers import TrainerCallback, Trainer
import torch

class SaveLoraPlusLayersCallback(TrainerCallback):
    def __init__(self, save_steps, layer_names, output_dir):
        self.save_steps = save_steps
        self.layer_names = layer_names
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.save_steps == 0:
            filtered_state_dict = {key: value for key, value in model.state_dict().items() if any(s in key for s in self.layer_names)}
            checkpoint_data = {
                "model_state_dict": filtered_state_dict,
                "optimizer_state_dict": kwargs['optimizer'].state_dict(),
                "scheduler_state_dict": kwargs['lr_scheduler'].state_dict(),
                "global_step": state.global_step,
                "epoch": state.epoch,
            }

            # Save the checkpoint
            torch.save(checkpoint_data, f"{self.output_dir}/checkpoint_{state.global_step}.pt")

def prepare_lora_plus_training(model):
    for n,p in model.named_parameters():
        p.requires_grad = False
        if any(s in n for s in ['lora', 'embed', 'norm']):
            p.requires_grad = True
            p.data = p.data.to(torch.float32)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable_params: {trainable_params:,}')


def split_dataset(dataset, train_size=.9):
    index = int(len(dataset) * train_size)
    return dataset.select(range(index)), dataset.select(range(index, len(dataset)))