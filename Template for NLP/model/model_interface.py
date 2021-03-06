import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from utils.model_util import diy_loss, diy_optimizer


class ModelInterface(pl.LightningModule):
    def __init__(self, kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
        self.load_model()
        self.configure_loss()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        loss += self.l1 * self.l1_norm() + self.l2 * self.l2_norm()
        return loss

    def validation_step(self, batch, batch_idx):
        # img, labels, filename = batch
        img, labels = batch
        out = self(img)
        loss = self.loss_function(out, labels)
        # label_digit = labels
        # out_digit = out.argmax(axis=-1)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        # correct_num = sum(label_digit == out_digit).cpu().item()
        return (loss.item(),)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    # def training_epoch_end(self, train_step_outputs):
    #     # outputs is a list of output from training_step
    #     loss = torch.stack([x for x in train_step_outputs]).mean()
    #     print(f'train_loss: {loss}')

    def validation_epoch_end(self, validation_step_outputs):
        # outputs is a list of output from validation_step
        # correct_num = sum([x[0] for x in validation_step_outputs])
        # total_num = sum([x[1] for x in validation_step_outputs])
        loss = sum([x[0] for x in validation_step_outputs]) / len(validation_step_outputs)
        # val_acc = correct_num / total_num

        self.log('val_loss', loss, on_epoch=True, prog_bar=False)
        # self.log('val_acc', val_acc, on_epoch=True, prog_bar=False)


    def l1_norm(self):
        return sum(p.abs().sum() for p in self.model.parameters() if p.ndim >= 2)

    def l2_norm(self):
        return sum((p ** 2).sum() for p in self.model.parameters() if p.ndim >= 2)

    def nuc_norm(self):
        return sum(torch.norm(p, p="nuc") for p in self.model.parameters() if p.ndim >= 2)

    def configure_optimizers(self):
        # optimizer
        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                            lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'diy':
            optimizer = diy_optimizer(self.model,
                                      lr=self.lr, weight_decay=self.weight_decay,
                                      epsilon=self.epsilon, momentum=self.momentum,
                                      correct_bias=self.correct_bias)
        else:
            raise ValueError("Unknown optimizer")

        # scheduler
        if self.lr_scheduler.lower() == 'cyclic':
            # TODO: add cyclic scheduler
            scheduler = lrs.CyclicLR(optimizer,
                                     base_lr=self.lr_decay_min_lr,
                                     max_lr=self.lr,
                                     step_size_up=self.opt_cfg["STEP_SIZE_UP"],
                                     step_size_down=self.opt_cfg["STEP_SIZE_DOWN"],
                                     mode=self.opt_cfg["MODE"])
        elif self.lr_scheduler.lower() == 'plateau':
            # TODO: add plateau scheduler
            scheduler = lrs.ReduceLROnPlateau(optimizer,
                                              mode=self.opt_cfg["MODE"],
                                              factor=self.opt_cfg["FACTOR"],
                                              patience=self.opt_cfg["PATIENCE"],
                                              verbose=self.opt_cfg["VERBOSE"])
        elif self.lr_scheduler.lower() == 'step':
            scheduler = lrs.StepLR(optimizer,
                                   step_size=self.lr_decay_step,
                                   gamma=self.lr_decay_rate)
        elif self.lr_scheduler.lower() == 'constant':
            scheduler = lrs.ConstantLR(optimizer)
        else:
            raise ValueError("Unknown scheduler")

        return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.loss
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
        name = self.model_name
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
