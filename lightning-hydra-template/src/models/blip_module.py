from typing import Any, Dict, Tuple
import json
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, SumMetric
from PIL import Image
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import numpy as np
import pandas as pd
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

        self.net = net
        self.valid_coco = COCO("../data/valid.json")
        self.test_result = {"img_name": [], "comments": []}

        self.train_loss = MeanMetric()
        self.score = SumMetric()
        self.score_best = MaxMetric()
        self.metric_weight = {
            "CIDEr": 4,
            "METEOR": 3,
            "Bleu_4": 1,
            "Bleu_3": 1,
            "ROUGE_L": 1,
        }
        self.result = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def on_train_start(self) -> None:
        pass

    def model_step(self, batch):
        img = [Image.open(os.path.join(i)) for i in batch["img"]]
        text = batch["text"]

        encoding = self.net.processor(
            images=img,
            text=text,
            padding="longest",
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        return encoding

    def valid_test_step(self, batch):
        image = [Image.open(os.path.join(i)) for i in batch["img"]]
        inputs = self.net.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        predict = self.net.model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
        )
        predict = self.net.processor.batch_decode(predict, skip_special_tokens=True)
        return predict

    def training_step(self, batch):
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        predict = self.valid_test_step(batch)

        self.result += [
            {"image_id": i, "caption": c}
            for i, c in zip(list(map(int, batch["img_id"])), predict)
        ]

    def on_validation_epoch_end(self) -> None:
        coco_result = self.valid_coco.loadRes(self.result)
        coco_eval = COCOEvalCap(self.valid_coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()

        valid = {}
        for metric, score in coco_eval.eval.items():
            valid[f"val/{metric}"] = score
            if metric in self.metric_weight.keys():
                self.score.update(score * self.metric_weight[metric])

        score = self.score.compute() / self.trainer.world_size
        self.score_best(score)

        score_best = self.score_best.compute()
        valid.update({"val/score": score, "val/best_score": score_best})
        self.log_dict(valid, sync_dist=True, prog_bar=True)

        self.result = []
        self.score.reset()

    def test_step(self, batch, batch_idx):
        predict = self.valid_test_step(batch)

        self.test_result["img_name"] += batch["img"]
        self.test_result["comments"] += predict

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        f = lambda x: x.split("/")[-1].split(".")[0]
        self.test_result["img_name"] = list(map(f, self.test_result["img_name"]))
        result = pd.DataFrame(self.test_result)
        test_csv = pd.read_csv("../data/test.csv")
        pd.merge(test_csv, result).to_csv("../test_caption.csv", index=False)

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
