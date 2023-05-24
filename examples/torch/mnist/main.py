import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from tqdm import tqdm, trange
import yaml

sys.path.append("../../..")
sys.path.append("../../../core/torch")
from core.torch.layer import Conv3dCubaLif
from core.torch.model import get_model, BaseModel


class Model(BaseModel):
    def __init__(self, e1, e2, p1):
        super().__init__()

        self.e1 = Conv3dCubaLif(**e1)
        self.e2 = Conv3dCubaLif(**e2)
        self.p1 = nn.LazyLinear(**p1)

    def forward(self, input):
        x = self.e1(input)
        x = self.e2(x)
        x = x.flatten(start_dim=1)
        x = self.p1(x)
        return x

    def reset(self):
        self.e1.reset()
        self.e2.reset()


def sequence(model, data, target):
    data = data.swapaxes(0, 1)  # (b, t, c, h, w) -> (t, b, c, h, w)
    target = target.unsqueeze(0).expand(len(data), -1)  # (b) -> (t, b)

    model.reset()
    loss = 0

    for x, y in zip(data, target):
        yhat = model(x)
        loss = loss + F.cross_entropy(yhat, y)

    loss = loss / len(data)
    accuracy = yhat.argmax(dim=1).eq(y).float().mean()
    return loss, accuracy


def train_epoch(model, dataloader, optimizer, device):
    model.train()

    epoch_loss = 0
    epoch_accuracy = 0
    passes = 0

    with tqdm(dataloader, desc="train batches", leave=False, dynamic_ncols=True) as batches_loop:
        for data, target in batches_loop:
            data, target = data.to(device), target.to(device)

            loss, accuracy = sequence(model, data, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            passes += 1

            batches_loop.set_postfix_str(
                f"train loss: {epoch_loss / passes:.4f}, train accuracy: {epoch_accuracy / passes * 100:.2f}"
            )

    return epoch_loss / passes, epoch_accuracy / passes


def eval_model(model, dataloader, device):
    model.eval()

    epoch_loss = 0
    epoch_accuracy = 0
    passes = 0

    with torch.no_grad():
        with tqdm(dataloader, desc="test batches", leave=False, dynamic_ncols=True) as batches_loop:
            for data, target in batches_loop:
                data, target = data.to(device), target.to(device)

                loss, accuracy = sequence(model, data, target)

                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                passes += 1

                batches_loop.set_postfix_str(
                    f"test loss: {epoch_loss / passes:.4f}, test accuracy: {epoch_accuracy / passes * 100:.2f}"
                )

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)


def main(config, args):
    # training params
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    device = config["training"]["device"]

    # dataset
    # make single image into a sequence repeating the image for the number of steps
    steps = config["dataset"]["steps"]
    x_to_bin = lambda x: x.gt(0.5).float()
    x_to_seq = lambda x: x.unsqueeze(0).expand(steps, -1, -1, -1)  # (c, h, w) -> (t, c, h, w)
    train_ds = MNIST("data", download=True, train=True, transform=Compose([ToTensor(), Lambda(x_to_bin), Lambda(x_to_seq)]))
    test_ds = MNIST("data", download=True, train=False, transform=Compose([ToTensor(), Lambda(x_to_bin), Lambda(x_to_seq)]))

    # dataloader
    train_loader = DataLoader(train_ds, shuffle=True, **config["dataloader"])
    test_loader = DataLoader(test_ds, shuffle=False, **config["dataloader"])

    # get model and trace it
    x, _ = next(iter(train_loader))
    model = get_model(Model, config["model"], data=x[:, 0], device=device)

    # logging with tensorboard
    if not args.debug:
        summary_writer = SummaryWriter()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    # loop that you can break
    try:
        with trange(epochs, desc="epochs", leave=False, dynamic_ncols=True) as epochs_loop:
            for t in epochs_loop:
                train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
                test_loss, test_accuracy = eval_model(model, test_loader, device)

                epochs_loop.set_postfix_str(
                    f"train loss: {train_loss:.4f}, train acc: {train_accuracy * 100:.2f}%, test loss: {test_loss:.4f},"
                    f" test acc: {test_accuracy * 100:.2f}%"
                )
                if not args.debug:
                    summary_writer.add_scalar("train_loss", train_loss, t)
                    summary_writer.add_scalar("train_accuracy", train_accuracy, t)
                    summary_writer.add_scalar("test_loss", test_loss, t)
                    summary_writer.add_scalar("test_accuracy", test_accuracy, t)

    except KeyboardInterrupt:
        pass

    if not args.debug:
        summary_writer.flush()
        summary_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config, args)
