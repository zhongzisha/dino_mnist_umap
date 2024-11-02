import sys,os,shutil
from torchvision import datasets
from PIL import Image


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result
   
PATH='/tmp/zhongz2'

# version 1, just resize to 256x256
if False:
    save_root = '/tmp/zhongz2/images'
    for subset in ['train', 'val']:
        for i in range(10):
            os.makedirs(os.path.join(save_root, subset, str(i)), exist_ok=True)

        dataset = datasets.MNIST(root=PATH, train=subset=='train', download=True)
        for idx, (img, label) in enumerate(dataset):
            img.resize((256, 256)).convert('RGB').save(os.path.join(save_root, subset, str(label), '{:05d}.jpg'.format(idx)))


# version 1, resize to 128x128, then, pad to 256x256 (didn't work), too much black margin
if False:
    save_root = '/tmp/zhongz2/images_v1'
    for subset in ['train', 'val']:
        for i in range(10):
            os.makedirs(os.path.join(save_root, subset, str(i)), exist_ok=True)

        dataset = datasets.MNIST(root=PATH, train=subset=='train', download=True)
        for idx, (img, label) in enumerate(dataset):
            img = img.resize((128, 128)).convert('RGB')
            img = add_margin(img, 64, 64, 64, 64, (0, 0, 0))
            img.save(os.path.join(save_root, subset, str(label), '{:05d}.jpg'.format(idx)))




# version 2, resize to 192x192, then, pad to 256x256
if False:
    save_root = '/tmp/zhongz2/images_v2'
    for subset in ['train', 'val']:
        for i in range(10):
            os.makedirs(os.path.join(save_root, subset, str(i)), exist_ok=True)

        dataset = datasets.MNIST(root=PATH, train=subset=='train', download=True)
        for idx, (img, label) in enumerate(dataset):
            img = img.resize((192, 192)).convert('RGB')
            img = add_margin(img, 32, 32, 32, 32, (0, 0, 0))
            img.save(os.path.join(save_root, subset, str(label), '{:05d}.jpg'.format(idx)))


if True:
    save_root = '/tmp/zhongz2/images_v3'
    for subset in ['train', 'val']:
        for i in range(10):
            os.makedirs(os.path.join(save_root, subset, str(i)), exist_ok=True)

        dataset = datasets.MNIST(root=PATH, train=subset=='train', download=True)
        for idx, (img, label) in enumerate(dataset):
            img = img.resize((160, 160)).convert('RGB')
            img = add_margin(img, 48, 48, 48, 48, (0, 0, 0))
            img.save(os.path.join(save_root, subset, str(label), '{:05d}.jpg'.format(idx)))




