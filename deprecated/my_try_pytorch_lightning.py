import torch
from torch import nn

import torchvision
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

ds_train = torchvision.datasets.MNIST(root="./minist/", train=True, download=True, transform=transform)
ds_valid = torchvision.datasets.MNIST(root="./minist/", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=4)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=8, shuffle=False, num_workers=4)

dl_train_iter = iter(dl_train)
inst = next(dl_train_iter)
print(len(ds_train))
print(len(ds_valid))

import pytorch_lightning as pl
import datetime


class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # 定义loss,以及可选的各种metrics
    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = nn.CrossEntropyLoss()(prediction, y)
        return loss

    # 定义optimizer,以及可选的lr_scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return {"test_loss": loss}


pl.seed_everything(1234)
model = Model()

ckpt_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min'
)

# gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
# gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
# tpus=1 则使用1个tpu训练

trainer = pl.Trainer(max_epochs=5, gpus=-1, callbacks=[ckpt_callback])

# 断点续训
# trainer = pl.Trainer(resume_from_checkpoint='./lightning_logs/version_31/checkpoints/epoch=02-val_loss=0.05.ckpt')

trainer.fit(model, dl_train, dl_valid)

