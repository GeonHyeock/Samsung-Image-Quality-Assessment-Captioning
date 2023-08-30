from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


class CocaModule(LightningModule):
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
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.train_loss = MeanMetric()
        self.cap_loss = MeanMetric()
        self.con_loss = MeanMetric()

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

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, text = batch["img"], batch["text"]
        text = pad_sequence(
            map(torch.tensor, self.tokenizer(text)["input_ids"]), batch_first=True
        ).to(self.device)
        return img, text

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        img, text = self.model_step(batch)
        caption_loss, contrastive_loss = self.net(text, img, return_loss=True)
        loss = caption_loss + contrastive_loss

        self.train_loss(loss)
        self.cap_loss(caption_loss)
        self.con_loss(contrastive_loss)
        self.log_dict(
            {
                "train/loss": self.train_loss,
                "train/caption_loss": self.cap_loss,
                "train/contrastive_loss": self.con_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # batch
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
        img, text = self.model_step(batch)
        logits = self.net(text=text, images=img)
        predict = self.tokenizer.batch_decode(logits.argmax(-1))
        target = [[i] for i in batch["text"]]

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
        )

    def on_validation_epoch_end(self) -> None:
        bleu3 = self.bleu3.compute()
        bleu4 = self.bleu4.compute()
        rouge = self.rouge.compute()

        # Score = CIDEr-D * 4 + METEOR * 3 + ((BLEU-4 + BLEU-3) / 2) * 2 + ROUGE-L * 1
        score = ((bleu3 + bleu4) / 2) + rouge["rougeL_fmeasure"]
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
