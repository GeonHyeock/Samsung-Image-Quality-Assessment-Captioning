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

        # for name, childs in self.model.named_children():
        #     if name == "vision_model":
        #         for n, param in childs.named_parameters():
        #             if ("layers.23" not in n) and ("post_layernorm" not in n):
        #                 param.requires_grad = False
        #             else:
        #                 print(f"train param : {n}")

        #     elif name == "text_decoder":
        #         for n, param in childs.named_parameters():
        #             if (
        #                 ("layer.11" not in n)
        #                 and ("cls" not in n)
        #                 and ("crossattention" not in n)
        #             ):
        #                 param.requires_grad = False
        #             else:
        #                 print(f"train param : {n}")

    def forward(self, **x):
        return self.model(**x)


if __name__ == "__main__":
    net = model("Salesforce/blip-image-captioning-large")
    import torch

    a = torch.load(
        "/home/user/captioning/lightning-hydra-template/logs/train/runs/2023-09-03_12-45-16/checkpoints/epoch_000.ckpt"
    )
    pass
