from typing import Any, Dict, Tuple
import json
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import numpy as np
import os


class BlipModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        model_dict = net.return_model()
        self.net = model_dict["model"]
        self.processor = model_dict["processor"]

        self.valid_coco = COCO("../data/valid.json")

        self.train_loss = MeanMetric()
        self.score_best = MaxMetric()
        self.result = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def on_train_start(self) -> None:
        pass

    def model_step(self, batch):
        img = [Image.open(os.path.join(i)) for i in batch["img"]]
        text = batch["text"]

        encoding = self.processor(
            images=img,
            text=text,
            padding="longest",
            return_tensors="pt",
        )
        # remove batch dimension
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        encoding.update({"img_name": list(map(int, batch["img_id"]))})
        return encoding

    def training_step(self, batch):
        batch = self.model_step(batch)
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")
        attention_mask = batch.pop("attention_mask")
        outputs = self.net(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids,
            attention_mask=attention_mask,
        )
        loss = outputs.loss

        self.train_loss(loss)
        self.log_dict(
            {
                "train/loss": self.train_loss,
                "train/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=len(input_ids),
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch = self.model_step(batch)
        predict = batch.pop("pixel_values")
        predict = self.net.generate(pixel_values=predict, max_length=50)
        predict = self.processor.batch_decode(predict, skip_special_tokens=True)
        self.result += [
            {"image_id": i, "caption": c} for i, c in zip(batch["img_name"], predict)
        ]

    def on_validation_epoch_end(self) -> None:
        coco_result = self.valid_coco.loadRes(self.result)
        coco_eval = COCOEvalCap(self.valid_coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()

        valid = {}
        for metric, score in coco_eval.eval.items():
            valid[f"val/{metric}"] = score

        # Score = CIDEr-D * 4 + METEOR * 3 + ((BLEU-4 + BLEU-3) / 2) * 2 + ROUGE-L * 1
        score = (
            valid["val/CIDEr"] * 4
            + valid["val/METEOR"] * 3
            + valid["val/Bleu_4"]
            + valid["val/Bleu_3"]
            + valid["val/ROUGE_L"]
        )

        self.score_best(score)
        valid.update({"val/score": score, "val/best_score": self.score_best.compute()})
        self.log_dict(valid, sync_dist=True, prog_bar=True)

        self.result = []

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/score",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
