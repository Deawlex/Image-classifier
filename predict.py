import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter
import random
import os
import seaborn as sns

import train
#Command Line Arguments

ap = argparse.ArgumentParser(description='predict-file')
ap.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
path = pa.checkpoint

def predict(path_image, model, top_k=5, power='gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    
    # No need for GPU on this part (just causes problems)
    if power == 'gpu':
        model.to("cuda")
    else:
        model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def process_image(path_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_pil = Image.open(path_image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
    
def imshow(path_image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = path_image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

train.load_checkpoint(path)

training_loader, testing_loader, validation_loader = train.load_data()

with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probs, labels, top_flowers = predict(path_image, model, number_of_outputs, power)

plt.figure(figsize = (6,10))
ax = plt.subplot(2,1,1)

# Set up title
flower_num = path_image.split('/')[2]
flower_name = cat_to_name[flower_num]

# Plot flower
img = process_image(path_image)
imshow(img, ax, title = flower_name);

# Plot bar chart
plt.subplot(2,1,2)
sns.barplot(x=probs, y=top_flowers, color=sns.color_palette()[0]);
plt.show()

print("Done!")