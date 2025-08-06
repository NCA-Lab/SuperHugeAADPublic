from logging import warning
import numpy as np
import torch


from .model_interface import MInterface
from ..linear import LinearABC


class LinearInterface(MInterface):
    model: LinearABC

    def identity_loss(self, *args):
        return self.fake_parameter * 0.0

    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)

        self.fake_parameter = torch.nn.Parameter(
            torch.zeros(1),
        )

        self.loss_fn = self.identity_loss

        self.post_model = torch.nn.Identity()

    def configure_optimizers(self):  # type: ignore
        return torch.optim.sgd.SGD(
            [
                self.fake_parameter,
            ],
            lr=0.0,
        )

    @property
    def lr(self):
        return 0.0

    @lr.setter
    def lr(self, value):
        warning(
            "Setting learning rate is not supported for LinearInterface. It is always 0.0. Skip setting learning rate."
        )

    def on_train_epoch_end(self):
        if not self.model._fitted:
            # Fit the model at the end of the training epoch. model.fit() may be called before training epoch ends.
            # E.g., lightning might go into one validation epoch just before one training epoch ends.
            self.model.fit()
        super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        if not self.model._fitted:
            # Fit the model at the start of the validation epoch
            self.model.fit()
        return super().on_validation_epoch_start()
