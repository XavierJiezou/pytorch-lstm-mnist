import sys
sys.path.append('.')
import os
import time
import copy
import torch
from torch import nn
import torch.optim as optim
from model.lstm import LSTM
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from script.split_dataset import split_dataset



class TrainOnMnist():
    def __init__(self, root: str, num_epochs: int, batch_size: int, input_size: int, time_step: int, lr: float):
        self.root = root
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.time_step = time_step
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.visualize_data()

    def split_dataset(self, transform=transforms.ToTensor()):
        dataset = split_dataset(self.root, transform)
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
        save_dir = './image/'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_data_mnist.png'), dpi=300, bbox_inches='tight')

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
        model = LSTM().to(self.device)
        return model

    def train_model(self):
        model = self.create_model()
        dataloader = self.make_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
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
            for inputs, labels in dataloader['train']:
                optimizer.zero_grad()
                inputs = inputs.squeeze(1).to(self.device)
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
            val_loss, val_acc = self.eval_model(model, dataloader['val'], criterion)
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
            f'BEST-VAL-ACC: {best_acc:.4f}'
        )
        model.load_state_dict(best_model_wts)
        return model
    
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
    
    def val_model():
        pass

    
    def test_model(self):
        pass


if __name__ == '__main__':
    TrainOnMnist(
        root='./data/mnist',
        num_epochs=3,
        batch_size=256,
        input_size=28,
        time_step=28,
        lr=1e-3
    ).train_model()
