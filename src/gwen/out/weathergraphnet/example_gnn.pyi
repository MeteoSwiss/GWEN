import torch
from _typeshed import Incomplete as Incomplete

parser: Incomplete
args: Incomplete
device: Incomplete
path: Incomplete
dataset: Incomplete
data: Incomplete
transform: Incomplete

class GCN(torch.nn.Module):
    conv1: Incomplete
    conv2: Incomplete
    def __init__(self, in_channels, hidden_channels, out_channels) -> None: ...
    def forward(self, x, edge_index, edge_weight: Incomplete | None = ...): ...

model: Incomplete
optimizer: Incomplete

def train() -> None: ...
def test() -> None: ...

best_val_acc: int
final_test_acc: int
times: Incomplete
start: Incomplete
loss: Incomplete
train_acc: Incomplete
val_acc: Incomplete
tmp_test_acc: Incomplete
best_val_acc = val_acc
test_acc = tmp_test_acc
