import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np

from models import *

learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'interpolated_adversarial_training'
mixup_alpha = 1.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

def mixup_data(x, y):
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        optimizer.zero_grad()

        benign_inputs, benign_targets_a, benign_targets_b, benign_lam = mixup_data(inputs, targets)
        benign_outputs = net(benign_inputs)
        loss1 = mixup_criterion(criterion, benign_outputs, benign_targets_a, benign_targets_b, benign_lam)
        benign_loss += loss1.item()

        _, predicted = benign_outputs.max(1)
        benign_correct += (benign_lam * predicted.eq(benign_targets_a).sum().float() + (1 - benign_lam) * predicted.eq(benign_targets_b).sum().float())
        if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign train loss:', loss1.item())

        adv = adversary.perturb(inputs, targets)
        adv_inputs, adv_targets_a, adv_targets_b, adv_lam = mixup_data(adv, targets)
        adv_outputs = net(adv_inputs)
        loss2 = mixup_criterion(criterion, adv_outputs, adv_targets_a, adv_targets_b, adv_lam)
        adv_loss += loss2.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += (adv_lam * predicted.eq(adv_targets_a).sum().float() + (1 - adv_lam) * predicted.eq(adv_targets_b).sum().float())
        if batch_idx % 10 == 0:
                print('Current adversarial train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial train loss:', loss2.item())

        loss = (loss1 + loss2) / 2
        loss.backward()
        optimizer.step()

    print('\nTotal benign train accuarcy:', 100. * benign_correct / total)
    print('Total adversarial train accuarcy:', 100. * adv_correct / total)
    print('Total benign train loss:', benign_loss)
    print('Total adversarial train loss:', adv_loss)

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current benign test loss:', loss.item())

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test(epoch)
