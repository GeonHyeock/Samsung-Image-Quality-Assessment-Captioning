from transformers import AutoProcessor, BlipForConditionalGeneration
from peft import inject_adapter_in_model, LoraConfig
import torch.nn as nn


class model(nn.Module):
    def __init__(self, pretrain):
        """
        text_decoder:
            bert:
                embeddings
                    embeddings
                    encoder
            cls:
                prediction:
                    transform
                    decoder
        """
        super(model, self).__init__()
        self.model = BlipForConditionalGeneration.from_pretrained(pretrain)
        self.processor = AutoProcessor.from_pretrained(pretrain)

        lora_target = [
            f"text_decoder.{name}"
            for name, module in self.model.text_decoder.named_modules()
            if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d))
        ]

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
    net = model("Salesforce/blip-image-captioning-large")
    pass
