"""The neural network model."""

import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Model(nn.Module):
    """If running on GPU, we should use different tensor datatypes.
    The following if/else loads the ResNet18 network in the right
    datatype, modifies the first convolution layer to grayscale, and
    sets the logit layer to multi-class output.
    """
    def __init__(self, config, n_classes, use_cuda=False):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.config = config

        self.base = models.resnet18(pretrained=config["use_pretrained"],
                                    progress=False)
        self.base = self.base.cuda() if use_cuda else self.base
        self.base.conv1 = nn.Conv2d(in_channels=1,
                                    out_channels=64,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)

        self.base.conv1 = self.base.conv1.cuda() if use_cuda else \
            self.base.conv1

        for base in self.base.parameters():
            base.requires_grad = config["trainable_resnet"]

        self.base.fc = nn.Linear(in_features=512,  # ResNet hardcode
                                 out_features=config["n_dense"][0])
        self.base.fc = self.base.fc.cuda() if use_cuda else self.base.fc

        add = 26 if config["metadata_filename"] else 0
        dense = nn.Linear(in_features=config["n_dense"][0] + add,
                          out_features=config["n_dense"][1])

        self.dense = dense.cuda() if use_cuda else dense

        logits = nn.Linear(in_features=config["n_dense"][1],
                           out_features=n_classes)

        self.logits = logits.cuda() if use_cuda else logits

        if config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(self.parameters(),
                                       lr=config["lr"],
                                       momentum=config["mmt"],
                                       nesterov=True,
                                       weight_decay=config["weight_decay"])
        elif config["optimizer"] == "AdamW":
            self.optimizer = optim.AdamW(self.parameters(),
                                         lr=config["lr"],
                                         weight_decay=config["weight_decay"])

        if config["scheduler"] == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                gamma=config["gamma"],
                step_size=config["gamma_step"])
        elif config["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=config["gamma"],
                patience=config["gamma_step"])

    def forward(self, image, data):
        x = F.relu(self.base(image))

        if self.config["metadata_filename"]:
            data = data.cuda() if self.use_cuda else data

        x = torch.cat((x, data), dim=1) if \
            self.config["metadata_filename"] else x
        x = F.relu(self.dense(x))
        x = self.logits(x)
        return x
