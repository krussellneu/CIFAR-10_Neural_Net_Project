# dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#################### Your Code ####################
ROOT_PATH='./cifar-10-batches-py'  # Modify this line with the path to the folder where folder "cifar-10-batches-py" locate
###################################################

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
            nn.Dropout(p=0.40, inplace=True),
            nn.Linear(32 * 32 * 3, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.40),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Dropout(p=0.40, inplace=True),
            nn.Linear(32 * 32 * 3, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.40),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


if __name__ == '__main__':
    best_mod = None
    results = list()
    model = NN().cuda()
    optimizer = optimizer(model.parameters(), momentum=momentum, lr=learn_rate)

    for epoch in range(50):
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
            torch.save(model.state_dict(), 'nn_cifar.pt')

        results.append(result)

    val_acc = [x['Accuracy'] for x in results]
    train_acc = [x['Train_Accuracy'] for x in results]
    plt.figure()
    plt.plot(train_acc, '-rx')
    plt.plot(val_acc, '-bx')
    plt.xlabel('Epochs by #')
    plt.ylabel('Accuracy by %')
    plt.legend(['Training', 'Validation'])
    plt.title('NN Accuracy Results')

    train_loss = [x.get('Train_Loss') for x in results]
    val_loss = [x['Loss'] for x in results]
    plt.figure()
    plt.plot(train_loss, '-bx')
    plt.plot(val_loss, '-rx')
    plt.xlabel('Epochs by #')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('NN Loss Results')
    plt.show()
