import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class ModelInterface(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
        self.load_model()
        self.configure_loss()
        # self.save_hyperparameters()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)["logits"]
        train_loss = self.loss_function(out, labels)
        out = torch.softmax(out, dim=-1)
        preds = torch.argmax(out, dim=-1)
        train_acc = accuracy(preds, labels)

        self.log('train_acc', train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)

        train_loss += self.l1 * self.l1_norm() + self.l2 * self.l2_norm()
        return train_loss

    def validation_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)["logits"]
        loss = self.loss_function(out, labels)
        out = torch.softmax(out, dim=-1)
        out_digit = out.argmax(axis=-1)
        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        correct_num = sum(labels == out_digit).cpu().detach()

        return (correct_num, len(out_digit), loss.detach())

    def test_step(self, batch, batch_idx):
        img, labels = batch
        out = self(img)["logits"]
        loss = self.loss_function(out, labels)
        out = torch.softmax(out, dim=-1)
        preds = torch.argmax(out, dim=-1)
        acc = accuracy(preds, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, labels = batch
        out = self(img)
        return out

    def validation_epoch_end(self, validation_step_outputs):
        # outputs is a list of output from validation_step
        correct_num = sum([x[0] for x in validation_step_outputs])
        total_num = sum([x[1] for x in validation_step_outputs])
        loss = sum([x[2] for x in validation_step_outputs]) / len(validation_step_outputs)
        val_acc = correct_num / total_num

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def l1_norm(self):
        return sum(p.abs().sum() for p in self.model.parameters() if p.ndim >= 2)

    def l2_norm(self):
        return sum((p ** 2).sum() for p in self.model.parameters() if p.ndim >= 2)

    def nuc_norm(self):
        return sum(torch.norm(p, p="nuc") for p in self.model.parameters() if p.ndim >= 2)

    def configure_optimizers(self):
        # optimizer
        if self.optimizer.lower() == 'adam':
            print(self.model)
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr, weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer")

        # scheduler
        if self.lr_scheduler.lower() == 'cyclic':
            # TODO: add cyclic scheduler
            scheduler = {"scheduler": lrs.CyclicLR(optimizer,
                                                   base_lr=self.lr_decay_min_lr,
                                                   max_lr=self.lr,
                                                   step_size_up=self.lr_decay_steps,
                                                   step_size_down=self.lr_decay_steps,
                                                   mode=self.lr_decay_mode),
                         "interval": "step"}
        elif self.lr_scheduler.lower() == 'cosine':
            scheduler = {"scheduler": lrs.CosineAnnealingLR(optimizer,
                                                            T_max=self.max_epochs),
                         "interval": "step"}
        elif self.lr_scheduler.lower() == 'plateau':
            # TODO: add plateau scheduler
            scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif self.lr_scheduler.lower() == 'step':
            scheduler = {"scheduler": lrs.StepLR(optimizer,
                                                 step_size=self.lr_decay_steps,
                                                 gamma=self.lr_decay_rate),
                         "interval": "step"}
        elif self.lr_scheduler.lower() == 'multistep':
            scheduler = {"scheduler": lrs.MultiStepLR(optimizer,
                                                      milestones=[135, 185],
                                                      gamma=self.lr_decay_rate),
                         "interval": "epoch"}
        elif self.lr_scheduler.lower() == 'one_cycle':
            scheduler = {"scheduler": lrs.OneCycleLR(optimizer,
                                                     max_lr=self.lr,
                                                     steps_per_epoch=self.lr_decay_steps,
                                                     epochs=self.max_epochs,
                                                     anneal_strategy='linear',
                                                     div_factor=self.max_epochs,
                                                     final_div_factor=self.max_epochs,
                                                     verbose=True
                                                     ),
                         "interval": "step"}
        elif self.lr_scheduler.lower() == 'cifar':
            steps_per_epoch = (50000 // self.train_batch_size) + 1

            scheduler = {"scheduler": lrs.OneCycleLR(optimizer,
                                                     max_lr=self.lr,
                                                     epochs=self.max_epochs,
                                                     steps_per_epoch=steps_per_epoch),
                         "interval": "step"}
        elif self.lr_scheduler.lower() == 'constant':
            scheduler = {"scheduler": lrs.ConstantLR(optimizer),
                         "interval": "step"}
        else:
            raise ValueError("Unknown scheduler")

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_loss(self):
        loss = self.loss
        if loss == 'mse':
            self.loss_function = torch.nn.MSELoss()
        elif loss == 'l1':
            self.loss_function = torch.nn.L1Loss()
        elif loss == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif loss == 'binary_cross_entropy':
            self.loss_function = torch.nn.BCELoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.model_name
        # add _data to the end of the name to match the name of the module
        name = name + "_net"
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.

        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return Model(**args1)
