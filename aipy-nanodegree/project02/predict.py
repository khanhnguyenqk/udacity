import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import os
import json

class predictor():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.resize_size = 256
        self.crop_size = 224
    
    def run(self):
        args = self.get_args()
        if args.gpu:
            if self.device == "cpu":
                raise Exception("Does not support gpu.")
        self.load_cat_name(args)
        self.load_model(args)
        top_p, top_c = self.predict(args.image_path, args)
        if self.cat_to_name:
            top_c = [self.cat_to_name[str(x)] for x in top_c]
        print(f"Top {args.top_k} prediction(s):")
        for p, c in zip(top_p, top_c):
            print(f"{c:30}{p}")
    
    def load_cat_name(self, args):
        if os.path.exists(args.category_names_fp):
            with open(args.category_names_fp, 'r') as f:
                self.cat_to_name = json.load(f)
                return
        self.cat_to_name = None

    def process_image(self, image:Image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        shortest = self.resize_size
        ratio = shortest / min(image.size)
        image = image.resize([int(x * ratio) for x in image.size])
        
        width, height = image.size 
        left = (width - self.crop_size)/2
        top = (height - self.crop_size)/2
        right = (width + self.crop_size)/2
        bottom = (height + self.crop_size)/2

        image = image.crop((left, top, right, bottom))
        
        np_image = np.array(image).astype(float)
        np_image /= 255
        mean = np.array(self.means)
        std = np.array(self.stds)

        np_image = (np_image - mean) / std
        np_image = np_image.transpose()
        
        return np_image

    def predict(self, image_path, args):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        print("Making prediction...")
        xs = torch.from_numpy(self.process_image(Image.open(image_path))).type(torch.FloatTensor)
        xs.unsqueeze_(0)
        xs = xs.to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            y_hats = self.model.forward(xs)
            self.model.train()

        ps = torch.exp(y_hats)
        top_p, top_c = ps.topk(args.top_k, dim=1)
        top_p, top_c = top_p.to("cpu").squeeze().numpy(), top_c.to("cpu").squeeze().numpy()
        return top_p, top_c
        
    def load_model(self, args):
        '''
        Load a checkpoint and rebuilds the model
        '''
        print(f"Loading model from checkpoint '{args.checkpoint_path}'. Device = {self.device}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)

        # Dynamically create model by name
        self.model = getattr(models, checkpoint['model_name'])(pretrained=True)
        named_params = list(self.model.named_parameters())
        self.classifier_name = named_params[-1][0].split('.')[0]
        # Dynamically set classifier
        setattr(self.model, self.classifier_name, checkpoint['classifier'])
        # Dynamically create optimizer
        self.optimizer = optim.Adam(getattr(self.model, self.classifier_name).parameters(), lr=checkpoint['lr'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epochs = checkpoint['epochs']

        # Freeze model params
        for param in self.model.parameters():
                param.requires_grad = False

        self.model.to(self.device)
    
    def get_args(self):
        """
        Get command line arguments
        """
        parser = argparse.ArgumentParser(description='Train a new network on a data set.')
        parser.add_argument('image_path', type=str, help='path to image')
        parser.add_argument('checkpoint_path', type=str, help='path to checkpoint')
        parser.add_argument('-t', '--top_k', metavar='top_k', type = int, default = 3, 
                        help = 'Top K most likely classes')
        parser.add_argument('-c', '--category_names_fp', metavar='category_names_fp', type = str, default = 'cat_to_name.json', 
                        help = 'Filepath of mapping of categories to real names')
        parser.add_argument('--gpu', action="store_true", help="Use GPU")
        return parser.parse_args()

if __name__ == "__main__":
    t = predictor()
    t.run()