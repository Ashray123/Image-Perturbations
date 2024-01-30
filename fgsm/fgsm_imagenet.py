


import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import argparse
import numpy as np
import cv2
from imagenet_labels import classes

parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default='images/goldfish.jpg', help='path to image')
parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], required=False, help="Which network?")
parser.add_argument('--y', type=int, required=False, help='Label')
parser.add_argument('--gpu', action="store_true", default=False)
args = parser.parse_args()

eps = 3.0

image_path = args.img
model_name = args.model
y_true = args.y
gpu = args.gpu

# Load image and preprocess
orig = cv2.imread(image_path)[..., ::-1]
height, width, _ = orig.shape  # Save original dimensions
img = cv2.resize(orig, (224, 224))
img = img.astype(np.float32)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)

# Load model
model = getattr(models, model_name)(pretrained=True)
model.eval()
criterion = nn.CrossEntropyLoss()

device = 'cuda' if gpu else 'cpu'

# Prediction before attack
inp = Variable(torch.from_numpy(img).to(device).float().unsqueeze(0), requires_grad=True)
out = model(inp)
pred = np.argmax(out.data.cpu().numpy())
print('Prediction before attack: %s' % (classes[pred].split(',')[0]))

# Run the attack
out = model(inp)
loss = criterion(out, Variable(torch.Tensor([float(pred)]).to(device).long()))
loss.backward()

# Perform FGSM attack
inp.data = inp.data + ((eps / 255.0) * torch.sign(inp.grad.data))
inp.grad.data.zero_()

# Predict on the adversarial image
pred_adv = np.argmax(model(inp).data.cpu().numpy())
print("After attack: eps [%f] \t%s" % (eps, classes[pred_adv].split(',')[0]))

# Deprocess image and return to original dimensions
adv = inp.data.cpu().numpy()[0]
perturbation = (adv - img).transpose(1, 2, 0)
adv = adv.transpose(1, 2, 0)
adv = (adv * std) + mean
adv = adv * 255.0
adv = adv[..., ::-1]
adv = np.clip(adv, 0, 255).astype(np.uint8)
perturbation = perturbation * 255
perturbation = np.clip(perturbation, 0, 255).astype(np.uint8)

# Resize to original dimensions
scale_factor_height = height / 224.0
scale_factor_width = width / 224.0
adv = cv2.resize(adv, (int(width), int(height)))
perturbation = cv2.resize(perturbation, (int(width), int(height)))

# Save images
cv2.imwrite('perturbation.png', perturbation)
cv2.imwrite('img_adv.png', adv)
print("Adversarial image and perturbation saved.")


