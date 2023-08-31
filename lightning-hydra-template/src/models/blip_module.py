from typing import Any, Dict, Tuple
import json
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore



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

        self.cider = None
        self.meteor = None
        self.bleu4 = BLEUScore(n_gram=4)
        self.bleu3 = BLEUScore(n_gram=3)
        self.rouge = ROUGEScore(rouge_keys="rougeL")

        self.score_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def on_train_start(self) -> None:
        self.bleu4.reset()
        self.bleu3.reset()
        self.rouge.reset()
        self.score_best.reset()

    def model_step(self, batch):
        pass
        

    def training_step(self, batch):
        input_ids = batch.pop("input_ids")
        pixel_values = batch.pop("pixel_values")
        attention_mask = batch.pop("attention_mask")
        outputs = self.net(
            input_ids=input_ids, pixel_values=pixel_values, labels=input_ids, attention_mask=attention_mask
        )
        loss = outputs.loss

        self.train_loss(loss)
        self.log_dict(
            {
                "train/loss": self.train_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(input_ids),
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
        predict = batch["pixel_values"]
        predict = self.net.generate(pixel_values=predict, max_length=50)
        predict = self.processor.batch_decode(predict, skip_special_tokens=True)
        target = list(map(lambda x: self.comments_dict[x], batch["img_name"]))

        self.bleu3(predict, target)
        self.bleu4(predict, target)
        self.rouge(predict, target)

        self.log_dict(
            {
                "val/bleu3": self.bleu3,
                "val/bleu4": self.bleu4,
                "val/rouge": self.rouge.rougeL_fmeasure[-1],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(predict),
        )

    def on_validation_epoch_end(self) -> None:
        bleu3 = self.bleu3.compute()
        bleu4 = self.bleu4.compute()
        rouge = self.rouge.compute()

        # Score = CIDEr-D * 4 + METEOR * 3 + ((BLEU-4 + BLEU-3) / 2) * 2 + ROUGE-L * 1
        score = bleu3 + bleu4 + rouge["rougeL_fmeasure"]
        self.score_best(score)
        self.log_dict(
            {"val/score": score, "val/best_score": self.score_best.compute()},
            sync_dist=True,
            prog_bar=True,
        )

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
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
