from typing import Any, Dict, Tuple
import json
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.meteor_score import meteor_score
from PIL import Image
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

        with open("../data/comments_dict.json", "r") as f:
            self.comments_dict = json.load(f)

        self.train_loss = MeanMetric()

        self.cider = MeanMetric()
        self.meteor = MeanMetric()
        self.bleu4 = BLEUScore(n_gram=4)
        self.bleu3 = BLEUScore(n_gram=3)
        self.rouge = ROUGEScore(rouge_keys="rougeL")

        self.compute_meteor = meteor_score
        self.score_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def on_train_start(self) -> None:
        self.bleu4.reset()
        self.bleu3.reset()
        self.rouge.reset()
        self.meteor.reset()
        self.score_best.reset()

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
        encoding.update({"img_name": batch["img_name"]})
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
        target = [
            {"image_id": i, "caption": c} for i, c in zip(batch["img_name"], predict)
        ]

    def on_validation_epoch_end(self) -> None:
        bleu3 = self.bleu3.compute()
        bleu4 = self.bleu4.compute()
        rouge = self.rouge.compute()
        meteor = self.meteor.compute()

        # Score = CIDEr-D * 4 + METEOR * 3 + ((BLEU-4 + BLEU-3) / 2) * 2 + ROUGE-L * 1
        score = meteor * 3 + bleu4 + bleu3 + rouge["rougeL_fmeasure"]
        self.score_best(score)
        self.log_dict(
            {
                "val/score": score,
                "val/best_score": self.score_best.compute(),
                "val/meteor": meteor,
                "val/belu3": bleu3,
                "val/belu4": bleu4,
                "val/rouge": rouge["rougeL_fmeasure"],
            },
            sync_dist=True,
            prog_bar=True,
        )

        self.bleu4.reset()
        self.bleu3.reset()
        self.rouge.reset()
        self.meteor.reset()
        self.score_best.reset()

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
