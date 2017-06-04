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
import os

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
    X_tensor = torch.cat(X)
    y_tensor = torch.LongTensor(y)
    model.eval()
    # dtype = torch.FloatTensor
    
    # Wrap the input tensors in Variables
    X_var = Variable(X_tensor, requires_grad=True)
    y_var = Variable(y_tensor)
    saliency = None
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
    #pickle.dump(saliency, open('saliency.pkl', 'wb'))
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
  test_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  
  class_to_idx = {'ReinierVinkeles': 6, 'WillemWitsen': 9, 'JohannesTavenraat': 4, 'GeorgeHendrikBreitner': 1,
                  'RembrandtHarmenszvanRijn': 7, 'BernardPicart': 0, 'SimonFokke': 8, 'IsaacIsraels': 2, 'JanLuyken': 3,
                  'MariusBauer': 5}

  model = torch.load("pytorch_full")
  model.eval()

  #filename=r'../data/images_top10/train/BernardPicart/en-NG-598-A.jpg'
  directory='../random_crop/r_crop_test/random_crop'
  labels = os.listdir(directory)
  files_and_labels = []
  for label in labels:
    for f in os.listdir(os.path.join(directory, label)):
      files_and_labels.append((os.path.join(directory, label, f), label))

  filenames, labels = zip(*files_and_labels)
  filenames = list(filenames)
  labels = list(labels)

  
  for i in range(len(filenames)):
    filename = filenames[i]
    label = labels[i]
    img = Image.open(filename).convert('RGB')
    X = test_transform(img).unsqueeze(0)
    y = class_to_idx[label]
    print filename, y
    image_id = os.path.splitext(os.path.basename(filename))[0]
    saliency = compute_saliency_maps([X], [y], model).numpy()
    pickle_filename = 'saliency_maps/' + image_id + '.pkl'
    pickle.dump(saliency, open(pickle_filename, 'wb'))
  
  #img = Image.open(filename).convert('RGB')
  #X = test_transform(img).unsqueeze(0)
  #print(X)
  #y = 0

  #show_saliency_maps([X], [y], model)
  # for x, y in train_loader:
  #   x_var = Variable(x.type(dtype))
  #   y_var = Variable(y.type(dtype).long())
  #   scores = model(x_var)
  #   #print(scores)
  #   _, preds = scores.data.cpu().max(1)
  #   print(preds[0][0], y[0])

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
