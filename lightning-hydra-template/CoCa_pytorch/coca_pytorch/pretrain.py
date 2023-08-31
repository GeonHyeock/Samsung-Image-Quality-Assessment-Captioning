from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import DataCollatorForLanguageModeling

vit = SimpleViT(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    patch_dropout=0.5,  # https://arxiv.org/abs/2212.00794
)

vit = Extractor(vit, return_embeddings_only=True, detach=False)


Pretrained = {
    "vit": vit,
    "tokenizer": AutoTokenizer.from_pretrained("bert-base-uncased"),
    "collactor": DataCollatorForLanguageModeling(
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        mlm_probability=0.15,
    ),
    "bert": BertModel.from_pretrained("bert-base-uncased"),
}
