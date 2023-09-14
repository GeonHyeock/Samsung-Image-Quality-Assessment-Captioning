from transformers import AutoProcessor, AutoModelForCausalLM
from peft import inject_adapter_in_model, LoraConfig
import torch.nn as nn


class model(nn.Module):
    def __init__(self, pretrain, train_module, lora_module):
        super(model, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(pretrain)
        self.processor = AutoProcessor.from_pretrained(pretrain)

        lora_target = [
            f"git.{name}"
            for name, module in self.model.git.named_modules()
            if ((isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d)))
            and any([m in name for m in lora_module])
            and not any([m in name for m in train_module])
        ]

        if lora_target:
            lora_config = LoraConfig(
                lora_alpha=4,
                lora_dropout=0.2,
                r=32,
                bias="lora_only",
                target_modules=lora_target,
            )
            self.model = inject_adapter_in_model(lora_config, self.model)

            for name, param in self.model.named_parameters():
                if any([m in name for m in train_module]):
                    param.requires_grad = True

            for p in self.model.output.parameters():
                p.requires_grad = True

        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                print(name)

    def forward(self, **x):
        return self.model(**x)


if __name__ == "__main__":
    net = model(
        "microsoft/git-large",
        ["visual_projection"],
        ["query", "value", "q_proj", "k_proj"],
    )

    pass
