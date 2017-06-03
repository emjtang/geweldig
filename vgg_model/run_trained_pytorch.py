from PIL import Image
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../data/images_top10/train')
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)

def main(args):
  # Figure out the datatype we will use; this will determine whether we run on
  # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor

  # Use the torchvision.transforms package to set up a transformation to use
  # for our images at training time. The train-time transform will incorporate
  # data augmentation and preprocessing. At training time we will perform the
  # following preprocessing on our images:
  # (1) Resize the image so its smaller side is 256 pixels long
  # (2) Take a random 224 x 224 crop to the scaled image
  # (3) Horizontally flip the image with probability 1/2
  # (4) Convert the image from a PIL Image to a Torch Tensor
  # (5) Normalize the image using the mean and variance of each color channel
  #     computed on the ImageNet dataset.
  train_transform = T.Compose([
    T.Scale(256),
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

  # You load data in PyTorch by first constructing a Dataset object which
  # knows how to load individual data points (images and labels) and apply a
  # transform. The Dataset object is then wrapped in a DataLoader, which iterates
  # over the Dataset to construct minibatches. The num_workers flag to the
  # DataLoader constructor is the number of background threads to use for loading
  # data; this allows dataloading to happen off the main thread. You can see the
  # definition for the base Dataset class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py
  #
  # and you can see the definition for the DataLoader class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L262
  #
  # The torchvision package provides an ImageFolder Dataset class which knows
  # how to read images off disk, where the image from each category are stored
  # in a subdirectory.
  #
  # You load data in PyTorch by first constructing a Dataset object which
  # knows how to load individual data points (images and labels) and apply a
  # transform. The Dataset object is then wrapped in a DataLoader, which iterates
  # over the Dataset to construct minibatches. The num_workers flag to the
  # DataLoader constructor is the number of background threads to use for loading
  # data; this allows dataloading to happen off the main thread. You can see the
  # definition for the base Dataset class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py
  #
  # and you can see the definition for the DataLoader class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L262
  #
  # The torchvision package provides an ImageFolder Dataset class which knows
  # how to read images off disk, where the image from each category are stored
  # in a subdirectory.
  #
  # You can read more about the ImageFolder class here:
  # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
  train_dset = ImageFolder(args.train_dir, transform=train_transform)
  train_loader = DataLoader(train_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)

  model = torch.load("pytorch_full")
  model.eval()
  #print(model)
  filename=r'../data/images_top10/train/BernardPicart/en-NG-598-A.jpg'
  for x, y in train_loader:
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())
    scores = model(x_var)
    #print(scores)
    _, preds = scores.data.cpu().max(1)
    print("predicted:", preds[0][0], ", actual", y[0])
  model.eval()
  img = Image.open(filename).convert('RGB')
  inputVar = Variable(train_transform(img).unsqueeze(0))
  prediction = model(inputVar)	 
  print("prediction:", prediction)

  probs, indices = (-nn.Softmax()(prediction).data).sort()
  print("probs", probs)
  print("indices", indices)
  probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
  print(probs)
  _, predicted = torch.max(prediction.data, 1)
  print("predicted:", predicted[0][0], ", actual", y[0])

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
