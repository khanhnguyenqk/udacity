import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import os

class trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_flower_class = 102
        self.batch_size = 64
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.checkpoint_fn = "checkpoint.pth"
    
    def run(self):
        args = self.get_args()
        if args.gpu:
            if self.device == "cpu":
                raise Exception("Does not support gpu.")
        self.create_dataloaders(args)
        self.create_model(args)
        self.train_model(args)
        self.save_checkpoint(args)

    def save_checkpoint(self, args):
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
            print(f"Created save directory: {args.save_dir}")

        self.model.class_to_idx = self.train_data.class_to_idx
        checkpoint = {'model_name':args.arch,
                    'epochs':args.epochs,
                    'batch_size':self.batch_size,
                    'classifier':self.classifier,
                    'optimizer':self.optimizer.state_dict(),
                    'state_dict':self.model.state_dict(),
                    'class_to_idx':self.model.class_to_idx,
                    'lr':args.learning_rate}
        checkpoint_fp = os.path.join(args.save_dir, self.checkpoint_fn)
        torch.save(checkpoint, checkpoint_fp)
        print(f"Checkpoint saved at: {checkpoint_fp}")

    def train_model(self, args):
        # train
        epochs = args.epochs
        steps = 0
        print_every_steps = 20
        running_loss = 0

        print(f"Start training. Epochs={epochs}. Device={self.device}...")

        for epoch in range(epochs):
            for xs, ys in self.trainloader:
                steps += 1

                xs, ys = xs.to(self.device), ys.to(self.device)
                self.optimizer.zero_grad()

                y_hats = self.model.forward(xs)
                loss = self.criterion(y_hats, ys)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every_steps:
                    continue

                test_loss = 0
                accuracy = 0
                self.model.eval()
                with torch.no_grad():
                    for xs, ys in self.validloader:
                        xs, ys = xs.to(self.device), ys.to(self.device)

                        y_hats = self.model.forward(xs)
                        test_loss += self.criterion(y_hats, ys)

                        ps = torch.exp(y_hats)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == ys.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every_steps:.3f}.. "
                f"Test loss: {test_loss/len(self.validloader):.3f}.. "
                f"Test accuracy: {accuracy/len(self.validloader):.3f}")
                running_loss = 0
                self.model.train()
                    
        print("Finish training...")

    def create_dataloaders(self, args):
        '''
        Create dataloaders
        '''
        train_dir = os.path.join(args.data_directory, 'train')
        valid_dir = os.path.join(args.data_directory, 'valid')
        test_dir = os.path.join(args.data_directory, 'test')
        # Define transforms for the training, validation, and testing sets
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.means, self.stds)])

        valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.means, self.stds)])

        # Load the datasets with ImageFolder
        self.train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        self.valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
        self.test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(self.valid_data, batch_size=self.batch_size)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
                
        print(f"Created train loader. Number of images = {len(self.train_data)}")
        print(f"Created validation loader. Number of images = {len(self.valid_data)}")
        print(f"Created test loader. Number of images = {len(self.test_data)}")

    def create_model(self, args):
        '''
        Dynamically create model based on model name
        '''
        print(f"Creating {args.arch} model. Learning rate = {args.learning_rate}..")
        # Dynamically create model based on model name
        try:
            self.model = getattr(models, args.arch)(pretrained=True)
            named_params = list(self.model.named_parameters())
            self.classifier_name = named_params[-1][0].split('.')[0]
            i = len(named_params) - 1
            while named_params[i][0].startswith(self.classifier_name):
                i -= 1
            self.classifier_in = list(named_params[i+1][1].shape)[1]
        except TypeError:
            raise Exception(f"Does not support model name '{args.arch}'")
        except AttributeError:
            raise Exception(f"Does not support model name '{args.arch}'")
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        # Create classifier
        self.classifier = nn.Sequential(nn.Linear(self.classifier_in, args.hidden_units),
                                nn.ReLU(),
                                nn.Dropout(p=0.2),
                                nn.Linear(args.hidden_units, self.num_flower_class),
                                nn.LogSoftmax(dim=1))
        
        setattr(self.model, self.classifier_name, self.classifier)

        self.model.to(self.device)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(getattr(self.model, self.classifier_name).parameters(), lr=args.learning_rate)

    def get_args(self):
        """
        Get command line arguments
        """
        parser = argparse.ArgumentParser(description='Train a new network on a data set.')
        parser.add_argument('data_directory', type=str, help='path to dataset director')
        parser.add_argument('-s', '--save_dir', metavar='save_dir', type = str, default = 'checkpoint', 
                        help = 'Set directory to save checkpoints')
        parser.add_argument('-a', '--arch', metavar='arch', type = str, default = 'vgg16', 
                        help = 'Set architechture')
        parser.add_argument('-l', '--learning_rate', metavar='learning_rate', type = float, default = 0.003, 
                        help = 'Set learning rate')
        parser.add_argument('-u', '--hidden_units', metavar='hidden_units', type = int, default = 512, 
                        help = 'Set hidden units')
        parser.add_argument('-e', '--epochs', metavar='epochs', type = int, default = 3, 
                        help = 'Set epochs')
        parser.add_argument('--gpu', action="store_true", help="Use GPU")
        return parser.parse_args()

if __name__ == "__main__":
    t = trainer()
    t.run()