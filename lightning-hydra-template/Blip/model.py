from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2Model
from peft import inject_adapter_in_model, LoraConfig
import torch.nn as nn
import torch


class model(nn.Module):
    def __init__(self, pretrain):
        super(model, self).__init__()
        self.model = Blip2Model.from_pretrained(pretrain)
        self.processor = Blip2Processor.from_pretrained(pretrain)

        lora_target = [
            f"qformer.{name}"
            for name, module in self.model.qformer.named_modules()
            if isinstance(module, nn.Linear)
        ] + ["language_projection"]

        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.2,
            r=64,
            bias="lora_only",
            target_modules=lora_target,
        )
        self.model = inject_adapter_in_model(lora_config, self.model)

        for i, v in self.model.named_parameters():
            if v.requires_grad == True:
                print(i)

    def forward(self, **x):
        return self.model(**x)


if __name__ == "__main__":
    net = model("Salesforce/blip2-opt-2.7b")
    pass
