import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class ModelInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()


    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, labels = batch
        out = self(img)["logits"]
        train_loss = self.loss_function(out, labels)
        out = torch.softmax(out, dim=-1)
        preds = torch.argmax(out, dim=-1)
        train_acc = accuracy(preds, labels)

        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)

        train_loss += self.hparams.l1 * self.l1_norm() + self.hparams.l2 * self.l2_norm()
        return train_loss

    def validation_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)["logits"]
        loss = self.loss_function(out, labels)
        out = torch.softmax(out, dim=-1)
        preds = torch.argmax(out, dim=-1)
        acc = accuracy(preds, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True,  prog_bar=True)

    def test_step(self, batch, batch_idx):
        img, labels = batch
        out = self(img)["logits"]
        loss = self.loss_function(out, labels)
        out = torch.softmax(out, dim=-1)
        preds = torch.argmax(out, dim=-1)
        acc = accuracy(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, labels = batch
        out = self(img)
        return out

    # def validation_epoch_end(self, validation_step_outputs):
    #     # outputs is a list of output from validation_step
    #     correct_num = sum([x[0] for x in validation_step_outputs])
    #     total_num = sum([x[1] for x in validation_step_outputs])
    #     loss = sum([x[2] for x in validation_step_outputs]) / len(validation_step_outputs)
    #     val_acc = correct_num / total_num
    #
    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def l1_norm(self):
        return sum(p.abs().sum() for p in self.model.parameters() if p.ndim >= 2)

    def l2_norm(self):
        return sum((p ** 2).sum() for p in self.model.parameters() if p.ndim >= 2)

    def nuc_norm(self):
        return sum(torch.norm(p, p="nuc") for p in self.model.parameters() if p.ndim >= 2)

    def configure_optimizers(self):
        # optimizer
        if self.hparams.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == 'diy':
            optimizer = diy_optimizer(self.model,
                                      lr=self.lr, weight_decay=self.hparams.weight_decay,
                                      epsilon=self.hparams.epsilon, momentum=self.hparams.momentum,
                                      correct_bias=self.hparams.correct_bias)
        else:
            raise ValueError("Unknown optimizer")

        # scheduler
        if self.hparams.lr_scheduler.lower() == 'cyclic':
            # TODO: add cyclic scheduler
            scheduler = {"scheduler": lrs.CyclicLR(optimizer,
                                                   base_lr=self.hparams.lr_decay_min_lr,
                                                   max_lr=self.hparams.lr,
                                                   step_size_up=self.hparams.lr_decay_steps,
                                                   step_size_down=self.hparams.lr_decay_steps,
                                                   mode=self.hparams.lr_decay_mode),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'cosine':
            max_steps = (50000//self.hparams.train_batch_size) * self.hparams.max_epochs
            scheduler = {"scheduler": lrs.CosineAnnealingLR(optimizer,
                                                            T_max=max_steps),  # self.hparams.max_epochs),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'plateau':
            # TODO: add plateau scheduler
            scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif self.hparams.lr_scheduler.lower() == 'step':
            scheduler = {"scheduler": lrs.StepLR(optimizer,
                                                 step_size=self.hparams.lr_decay_steps,
                                                 gamma=self.hparams.lr_decay_rate),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'multistep':
            scheduler = {"scheduler": lrs.MultiStepLR(optimizer,
                                                      milestones=[135, 185],
                                                      gamma=self.hparams.lr_decay_rate),
                         "interval": "epoch"}
        elif self.hparams.lr_scheduler.lower() == 'one_cycle':
            scheduler = {"scheduler": lrs.OneCycleLR(optimizer,
                                                     max_lr=self.hparams.lr,
                                                     steps_per_epoch=self.hparams.lr_decay_steps,
                                                     epochs=self.hparams.max_epochs,
                                                     anneal_strategy='linear',
                                                     div_factor=self.hparams.max_epochs,
                                                     final_div_factor=self.hparams.max_epochs,
                                                     verbose=True
                                                     ),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'cifar':
            steps_per_epoch = (50000 // self.hparams.train_batch_size) + 1

            scheduler = {"scheduler": lrs.OneCycleLR(optimizer,
                                                     max_lr=self.hparams.lr,
                                                     epochs=self.hparams.max_epochs,
                                                     steps_per_epoch=steps_per_epoch,
                                                     ),
                         "interval": "step"}
        elif self.hparams.lr_scheduler.lower() == 'constant':
            scheduler = {"scheduler": lrs.ConstantLR(optimizer),
                         "interval": "step"}
        else:
            raise ValueError("Unknown scheduler")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_loss(self):
        loss = self.hparams.loss
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'cross_entropy':
            self.loss_function = F.cross_entropy
        elif loss == 'binary_cross_entropy':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'diy':
            # calculate loss in the model class
            self.loss_function = diy_loss
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # add _data to the end of the name to match the name of the module
        name = name + "_net"
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.

        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except ImportError:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
