import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import matplotlib.image as image
import numpy as np
import os

epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'basic_training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

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
            x = x.detach() - alpha * torch.sign(grad.detach())
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
checkpoint = torch.load('./checkpoint/' + file_name)
net.load_state_dict(checkpoint['net'])

adversary = LinfPGDAttack(net)

num_classes = 10
number_per_class = {}

for i in range(num_classes):
    number_per_class[i] = 0

def custom_imsave(img, label):
    path = 'custom_dataset/' + str(label) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    image.imsave(path + str(number_per_class[label]) + '.jpg', img)
    number_per_class[label] += 1

def process():
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = (targets + 1) % 10
        adv = adversary.perturb(inputs, targets)

        print("[ Current Batch Index: " + str(batch_idx) + " ]")
        for i in range(inputs.size(0)):
            custom_imsave(adv.cpu()[i], targets.cpu()[i].item())

process()
