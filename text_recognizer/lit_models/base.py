import argparse
import pytorch_lightning as pl
import torch
import numpy as np


OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class Accuracy(pl.metrics.Accuracy):
    """Accuracy Metric with a hack."""

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Metrics in Pytorch-lightning 1.2+ versions expect preds to be between 0 and 1 else fails with the ValueError:
        "The `preds` should be probabilities, but values were detected outside of [0,1] range."
        This is being tracked as a bug in https://github.com/PyTorchLightning/metrics/issues/60.
        This method just hacks around it by normalizing preds before passing it in.
        Normalized preds are not necessary for accuracy computation as we just care about argmax().
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr", LR)

        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)
        '''
        if loss not in ("ctc", "transformer"):
            self.loss_fn = getattr(torch.nn.functional, loss)
        '''    

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.predictions=np.array([])
        self.train_size=0

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss,on_step=False, on_epoch=True,prog_bar=False)
        self.train_acc(logits, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        #print('validating ')
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True,prog_bar=False)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False)

    def reset_predictions(self):
        print('\nResetting Predictions\n')
        self.predictions=np.array([]) 
        self.total_predictions=np.array([])   

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        
        preds = torch.nn.functional.softmax(logits, dim=-1)
        #print('prd shape 0',self.predictions.shape[0])
        if self.predictions.shape[0]==0:
            self.predictions=preds.cpu().detach().numpy()
        else:    
            np.vstack([self.predictions,preds.cpu().detach().numpy()])

        self.test_acc(logits, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_batch_end(self,outputs, batch, batch_idx, dataloader_idx):
        print('test batch end ')
        if self.total_predictions.shape[0]==0:
            self.total_predictions=self.predictions
        else:    
            np.vstack([self.total_predictions,self.predictions])

    def on_test_epoch_end(self,outputs):
        print('test epoch end ', type(outputs))        
    
    def on_train_epoch_start(self):
        print('epoch start')
        self.train_size=0
        print('train size=',self.train_size)

    def on_train_batch_end(self,outputs):
        self.train_size +=len(outputs)
        print('train batch end , train size =', self.train_size)

    def training_epoch_end(self, outputs):
        ts=self.train_size
        self.log("train_set_size",ts)
        
    '''
    def validation_epoch_end(self, outputs):
        #print('validation outputs ',outputs)

    def test_epoch_end(self, outputs):
        #print('test outputs ',outputs)    
    '''    

