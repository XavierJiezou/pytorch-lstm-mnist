import sys
sys.path.append('.')
import os
import time
import copy
import yaml
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from model.lstm import LSTM
from easydict import EasyDict
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from script.split_dataset import split_dataset



class TrainOnMnist():
    def __init__(self, config_fname):
        self.parse_args(EasyDict(yaml.load(open(config_fname), yaml.FullLoader)))
    
    def parse_args(self, args):
        # data options
        self.data_root = args.data.data_root
        self.train_ratio = args.data.train_ratio
        self.val_ratio = args.data.val_ratio
        self.batch_size = args.data.batch_size
        self.visualize_save = args.data.visualize_save

        # model options
        self.input_size = args.model.input_size
        self.hidden_size = args.model.hidden_size
        self.num_layers = args.model.num_layers
        self.output_size = args.model.output_size

        # train options
        self.num_epochs = args.train.num_epochs
        self.sequence_length = args.train.sequence_length
        self.learning_rate = args.train.learning_rate
        self.device = args.train.device if torch.cuda.is_available() else 'cpu'
        self.save_path = args.train.save_path

    def split_dataset(self, transform=transforms.ToTensor()):
        dataset = split_dataset(self.data_root, transform, self.train_ratio, self.val_ratio)
        return dataset

    def visualize_data(self):
        dataset = self.split_dataset()
        imgs_dict = {i: [] for i in range(10)}
        for image, label in dataset['train']:
            if len(imgs_dict[label]) < 10:
                imgs_dict[label] += [image]
            else:
                if sum([len(item) for item in imgs_dict.values()]) == 100:
                    break
                else:
                    continue
        i = 1
        for image_list in imgs_dict.values():
            for image in image_list:
                plt.subplot(10, 10, i)
                plt.imshow(image.permute(1, 2, 0), cmap='gray')
                plt.axis('off')
                i+=1
        os.makedirs(os.path.split(self.visualize_save)[0], exist_ok=True)
        plt.savefig(self.visualize_save, dpi=300, bbox_inches='tight')
        print(f'The visualization result has been saved in `{self.visualize_save}`')

    def make_dataloader(self):
        dataset = self.split_dataset()
        dataloader = {
            split: DataLoader(
                dataset=dataset[split],
                batch_size=self.batch_size,
                shuffle=True
            ) for split in dataset
        }
        return dataloader

    def create_model(self):
        model = LSTM(self.input_size, self.hidden_size, self.num_layers, self.output_size).to(self.device)
        return model

    def train_model(self, model, dataloader, optimizer, scheduler, criterion):
        start_time = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            model.train()
            start_epoch_time = time.time()
            lr = optimizer.param_groups[0]['lr']
            print(
                f'EPOCH: {epoch+1:0>{len(str(self.num_epochs))}}/{self.num_epochs}',
                f'LR: {lr:.4f}',
                end=' '
            )
            train_loss = 0.0
            train_corrects = 0
            for inputs, labels in tqdm(dataloader['train']):
                optimizer.zero_grad()
                inputs = inputs.reshape(inputs.size(0), self.sequence_length, self.input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_corrects += (outputs.argmax(1) == labels).sum()
            scheduler.step()
            train_loss /= len(dataloader['train'].dataset)
            train_acc = train_corrects / len(dataloader['train'].dataset)
            val_loss, val_acc = self.val_model(model, dataloader['val'], criterion)
            end_epoch_time = time.time()
            total_epoch_time = end_epoch_time - start_epoch_time
            print(
                f'TRAIN_LOSS: {train_loss:.4f}',
                f'TRAIN_ACC: {train_acc:.4f} ',
                f'VAL-LOSS: {val_loss:.4f}',
                f'VAL-ACC: {val_acc:.4f} ',
                f'EPOCH-TIME: {total_epoch_time//60:.0f}m{total_epoch_time%60:.0f}s',
                end='\n'
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                pass
        end_time = time.time()
        total_time = end_time-start_time
        print('-'*10)
        print(
            f'TOTAL-TIME: {total_time//60:.0f}m{total_time%60:.0f}s',
            f'BEST-VAL-ACC: {best_acc:.4f}',
            end=' '
        )
        return best_model_wts
    
    def eval_model(self, model, dataloader, criterion):
        model.eval()
        eval_loss = 0.0
        eval_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.squeeze(1).to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = model(inputs)
            eval_loss += criterion(outputs, labels).item() * inputs.size(0)
            eval_corrects += (outputs.argmax(1) == labels).sum()
        eval_loss /= len(dataloader.dataset)
        eval_acc = eval_corrects / len(dataloader.dataset)
        return eval_loss, eval_acc
    
    def val_model(self, model, dataloader, criterion):
        val_loss, val_acc = self.eval_model(model, dataloader, criterion)
        return val_loss, val_acc
        
    def test_model(self, model, dataloader, criterion):
        test_loss, test_acc = self.eval_model(model, dataloader, criterion)
        return test_loss, test_acc
    
    def save_model(self, model_wts, save_path):
        torch.save(model_wts, save_path)

    def __main__(self):
        model = self.create_model()
        dataloader = self.make_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.num_epochs//2, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_model_wts = self.train_model(model, dataloader, optimizer, scheduler, criterion)
        model.load_state_dict(best_model_wts)
        test_loss, test_acc = self.test_model(model, dataloader['test'], criterion)
        print(
            f'TEST-LOSS: {test_loss:.4f}',
            f'TEST-ACC: {test_acc:.4f} ',
        )
        self.save_model(best_model_wts, self.save_path)

if __name__ == '__main__':
    TrainOnMnist('./config/mnist.yaml').__main__()
