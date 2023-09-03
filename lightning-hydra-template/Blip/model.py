from transformers import AutoProcessor, BlipForConditionalGeneration
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

        for name, childs in self.model.named_children():
            if name == "vision_model":
                for param in childs.parameters():
                    param.requires_grad = False

            elif name == "text_decoder":
                for param in childs.bert.parameters():
                    param.requires_grad = False

    def forward(self, **x):
        return self.model(**x)


if __name__ == "__main__":
    net = model("Salesforce/blip-image-captioning-base")
    import torch

    a = torch.load(
        "/home/user/captioning/lightning-hydra-template/logs/train/runs/2023-09-03_12-45-16/checkpoints/epoch_000.ckpt"
    )
    pass
