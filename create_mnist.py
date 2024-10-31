import sys,os,shutil
from torchvision import datasets

PATH='./'
save_root = './images'
for subset in ['train', 'val']:
    for i in range(10):
        os.makedirs(os.path.join(save_root, subset, str(i)), exist_ok=True)

    dataset = datasets.MNIST(root=PATH, train=subset=='train', download=True)
    for idx, (img, label) in enumerate(dataset):
        img.resize((256, 256)).convert('RGB').save(os.path.join(save_root, subset, str(label), '{:05d}.jpg'.format(idx)))




