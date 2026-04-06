import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
from vggnet import VGGNet
from load_cifar10 import train_loader,test_loader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch_num = 200
    lr = 0.001
    net = VGGNet().to(device)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)

    for epoch in range(epoch_num):
        print("epoch:", epoch)
        net.train()

        for i, data in enumerate(train_loader):
            inputs, label = data

            inputs, labels = inputs.to(device), label.to(device)

            output = net(inputs)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            print("loss:", loss.item())

    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(net.state_dict(), "./models/{}".format(epoch + 1))
    scheduler.step()




if __name__ == '__main__':
    main()




