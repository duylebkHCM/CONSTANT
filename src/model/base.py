from torch import nn
from collections import namedtuple

Loss_Storage = namedtuple("Loss_Storage", "loss_value loss_weight")


class BaseModule(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def _ouput_loss(self, loss_value):
        return Loss_Storage(loss_value=loss_value, loss_weight=self.loss_weight)
