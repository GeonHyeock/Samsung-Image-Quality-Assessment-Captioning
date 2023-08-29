from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
from vit_pytorch.extractor import Extractor

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

Img_encoder_dict = {"vit": vit}
