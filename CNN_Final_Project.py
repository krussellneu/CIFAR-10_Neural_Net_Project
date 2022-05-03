import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

ROOT_PATH='./cifar-10-batches-py'

BATCH_SIZE = 128
momentum = 0.9
learn_rate = 0.001
optimizer = torch.optim.SGD

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = CIFAR10(root=ROOT_PATH, download=True, train=True, transform=transform)
eval_dataset = CIFAR10(root=ROOT_PATH, train=False, transform=transform)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


@torch.no_grad()
def model_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))


class NNBase(nn.Module):
    def train_step(self, images, labels):
        out = self(images)
        return {'Loss': F.cross_entropy(out, labels),
                'Accuracy': model_accuracy(out, labels)}

    def valid_step(self, images, labels):
        out = self(images)
        return {'Loss': F.cross_entropy(out, labels).detach(),
                'Accuracy': model_accuracy(out, labels)}

    @staticmethod
    def valid_epoch_end(outputs):
        losses = torch.stack([x['Loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['Accuracy'] for x in outputs]).mean()
        return {'Loss': losses.item(), 'Accuracy': accuracy.item()}

    @staticmethod
    def print_results(ep, res):
        print('Epoch: [{0}] => Train Accuracy: {1:.2f}%,'.format(
            ep + 1,
            res['Train_Accuracy'] * 100),
            'Valid Accuracy: {0:.2f}%, Train Loss: {1:.4f},'.format(
                res['Accuracy'] * 100, res['Train_Loss']),
            'Valid Loss: {0:.4f}'.format(res['Loss']))


class NN(NNBase):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),  # (b x 3 x 32 x 32) depth-wise
            nn.Conv2d(3, 64, kernel_size=1),  # (b x 64 x 32 x 32)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(64),  # Normalize channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # (b x 64 x 32 x 32) depth-wise
            nn.Conv2d(64, 128, kernel_size=1),  # (b x 128 x 32 x 32)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(128),  # Normalize channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool with 2x2 filter and 2 stride (b x 128 x 16 x 16)
            nn.Dropout(0.25),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # (b x 128 x 16 x 16) depth-wise
            nn.Conv2d(128, 256, kernel_size=1),  # (b x 256 x 16 x 16)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(256),  # Normalize channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),  # (b x 256 x 16 x 16) depth-wise
            nn.Conv2d(256, 256, kernel_size=1),  # (b x 256 x 16 x 16)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(256),  # Normalize channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool with 2x2 filter and 2 stride (b x 256 x 8 x 8)
            nn.Dropout(0.25),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),  # (b x 256 x 8 x 8) depth-wise
            nn.Conv2d(256, 512, kernel_size=1),  # (b x 512 x 8 x 8)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(512),  # Normalize channels
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),  # (b x 512 x 8 x 8) depth-wise
            nn.Conv2d(512, 512, kernel_size=1),  # (b x 512 x 8 x 8)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(512),  # Normalize channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool with 2x2 filter and 2 stride (b x 512 x 4 x 4)
            nn.Dropout(0.25),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),  # (b x 512 x 4 x 4) depth-wise
            nn.Conv2d(512, 1024, kernel_size=1),  # (b x 1024 x 4 x 4)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, groups=1024),  # (b x 1024 x 4 x 4) depth-wise
            nn.Conv2d(1024, 1024, kernel_size=1),  # (b x 1024 x 4 x 4)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(1024),  # Normalize channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool with 2x2 filter and 2 stride (b x 1024 x 2 x 2)
            nn.Dropout(0.25),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, groups=1024),  # (b x 1024 x 2 x 2) depth-wise
            nn.Conv2d(1024, 512, kernel_size=1),  # (b x 512 x 2 x 2)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(512),  # Normalize channels
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),  # (b x 512 x 2 x 2) depth-wise
            nn.Conv2d(512, 512, kernel_size=1),  # (b x 512 x 2 x 2)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(512),  # Normalize channels
            nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512),  # (b x 512 x 2 x 2) depth-wise
            nn.Conv2d(512, 512, kernel_size=1),  # (b x 256 x 2 x 2)
            nn.ReLU(),  # Call Relu activation function
            nn.BatchNorm2d(512),  # Normalize channels
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool with 2x2 filter and 2 stride (b x 256 x 1 x 1)
            nn.Dropout(0.25)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.40, inplace=True),
            nn.Linear(512 * 1 * 1, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.40),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    best_mod = None
    results = list()
    model = NN().cuda()

    optimizer = optimizer(model.parameters(), momentum=momentum, lr=learn_rate)

    for epoch in range(100):
        train_loss = list()
        train_acc = list()
        outputs = list()

        model.train()

        for input, target in train_data_loader:
            input = input.cuda()
            target = target.cuda()
            result = model.train_step(input, target)
            train_loss.append(result['Loss'])
            train_acc.append(result['Accuracy'])
            optimizer.zero_grad()
            result['Loss'].backward()
            optimizer.step()

        model.eval()

        for input, target in eval_data_loader:
            input = input.cuda()
            target = target.cuda()
            output = model.valid_step(input, target)
            outputs.append(output)

        result = model.valid_epoch_end(outputs)
        result['Train_Loss'] = torch.stack(train_loss).mean().item()
        result['Train_Accuracy'] = torch.stack(train_acc).mean().item()
        model.print_results(epoch, result)

        if best_mod is None or best_mod < result['Accuracy']:
            best_mod = result['Accuracy']
            torch.save(model.state_dict(), 'cnn_cifar.pt')

        results.append(result)

    val_acc = [x['Accuracy'] for x in results]
    train_acc = [x['Train_Accuracy'] for x in results]
    plt.figure()
    plt.plot(train_acc, '-rx')
    plt.plot(val_acc, '-bx')
    plt.xlabel('Epochs by #')
    plt.ylabel('Accuracy by %')
    plt.legend(['Training', 'Validation'])
    plt.title('CNN Accuracy Results')

    train_loss = [x.get('Train_Loss') for x in results]
    val_loss = [x['Loss'] for x in results]
    plt.figure()
    plt.plot(train_loss, '-bx')
    plt.plot(val_loss, '-rx')
    plt.xlabel('Epochs by #')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('CNN Loss Results')
    plt.show()
