'''
Compute saliency maps using PyTorch because Tensorflow SUCKS
'''
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pickle

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

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    # dtype = torch.FloatTensor
    
    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with a backward pass.               #
    ##############################################################################
    # model.type(dtype)

    scores = model(X_var)

    scores_arr = scores.gather(1, y_var.view(-1, 1)).squeeze()
    sum_scores = torch.sum(scores_arr)

    sum_scores.backward()
    
    saliency = X_var.grad.data.abs().max(dim=1)[0].squeeze()
    
    return saliency
def show_saliency_maps(X, y, model):
    # Convert X and y from numpy arrays to Torch Tensors
    # X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X])
    X_tensor = torch.cat(X)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    pickle.dump(saliency, open('saliency.pkl', 'wb'))
    #N = X.shape[0]
    N = 1
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

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
  test_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
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
  model = torch.load("pytorch_full")
  model.eval()

  filename=r'../data/images_top10/train/BernardPicart/en-NG-598-A.jpg'



  model = torch.load("pytorch_full")
  #print(model)

  img = Image.open(filename).convert('RGB')
  X = test_transform(img).unsqueeze(0)
  pickle.dump(X, open('X.pkl', 'wb'))
  y = 0
  show_saliency_maps([X], [y], model)
  # for x, y in train_loader:
  #   x_var = Variable(x.type(dtype))
  #   y_var = Variable(y.type(dtype).long())
  #   scores = model(x_var)
  #   #print(scores)
  #   _, preds = scores.data.cpu().max(1)
  #   print(preds[0][0], y[0])
   
#  img = skimage.io.imread(filename)
#  x = V(centre_crop(img).unsqueeze(0), volatile=True)
#  logit = model(x)
#  print(logit)
  #model = pytorch_full

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
