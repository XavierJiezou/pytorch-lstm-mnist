import sys
sys.path.append('.')
import os
import re
import time
import copy
import yaml
import torch
from torch import nn
from tqdm import tqdm
from loguru import logger
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
        self.get_logger(logger)
        self.visualize_data()
        self.__main__()
        self.visualize_log()

    def parse_args(self, args):
        # data options
        self.data_root = args.data.data_root
        self.train_ratio = args.data.train_ratio
        self.val_ratio = args.data.val_ratio
        self.batch_size = args.data.batch_size
        self.visualize_data_save = args.data.visualize_data_save

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

        # log options
        self.sink = args.log.sink
        self.level = args.log.level
        self.format = args.log.format
        self.visualize_log_save = args.log.visualize_log_save

    def get_logger(self, logger):
        self.logger = logger
        self.logger.remove()
        self.logger.add(sink=self.sink, level=self.level, format=self.format)
        self.logger.add(sink=sys.stderr, level=self.level, format=self.format)

    def split_dataset(self, transform=transforms.ToTensor()):
        dataset = split_dataset(
            self.data_root, 
            transform,
            self.train_ratio, 
            self.val_ratio
        )
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
                i += 1
        os.makedirs(os.path.split(self.visualize_data_save)[0], exist_ok=True)
        plt.savefig(self.visualize_data_save, dpi=300, bbox_inches='tight')
        self.logger.info(f'The visualization data has been saved in `{self.visualize_data_save}`')
    
    def visualize_log(self):
        result_dict = {
            'Loss': {
                'TRAIN-LOSS': [],
                'VAL-LOSS': []
            },
            'Accuracy': {
                'TRAIN-ACC': [],
                'VAL-ACC': []
            }
        }
        with open(self.sink) as f:
            for line in f.readlines():
                try:
                    for i in result_dict:
                        for j in result_dict[i]:
                            result_dict[i][j] += [float(re.findall(f'{j}: (.*?) ', line)[0])]
                except:
                    pass

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(result_dict['Loss']['TRAIN-LOSS'], label='Train-loss')
        ax1.plot(result_dict['Loss']['VAL-LOSS'], label='Val-loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(result_dict['Accuracy']['TRAIN-ACC'], label='Train-accuracy', linestyle='--')
        ax2.plot(result_dict['Accuracy']['VAL-ACC'], label='Val-accuracy', linestyle='--')
        ax2.set_ylabel('Accuracy')

        fig.legend(loc=1, bbox_to_anchor=(1, 0.8), bbox_transform=ax1.transAxes)

        os.makedirs(os.path.split(self.visualize_log_save)[0], exist_ok=True)
        plt.savefig(self.visualize_log_save, dpi=300, bbox_inches='tight')
        self.logger.info(f'The visualization result has been saved in `{self.visualize_log_save}`')

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
        model = LSTM(
            self.input_size, 
            self.hidden_size,
            self.num_layers, 
            self.output_size
        ).to(self.device)
        return model

    def train_model(self, model, dataloader, optimizer, scheduler, criterion):
        start_time = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0
        for epoch in range(self.num_epochs):
            model.train()
            start_epoch_time = time.time()
            train_loss = 0.0
            train_corrects = 0
            epoch_print_str = f'EPOCH: {epoch+1:0>{len(str(self.num_epochs))}}/{self.num_epochs}'
            lr_print_str = f'LR: {optimizer.param_groups[0]["lr"]:.4f}'
            for inputs, labels in tqdm(dataloader['train'], desc=f'{epoch_print_str} STEP'):
                optimizer.zero_grad()
                inputs = inputs.reshape(
                    inputs.size(0), 
                    self.sequence_length, 
                    self.input_size
                ).to(self.device)
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
            self.logger.info(
                ' '.join([
                    epoch_print_str,
                    lr_print_str,
                    f'TRAIN-LOSS: {train_loss:.4f}',
                    f'TRAIN-ACC: {train_acc:.4f} ',
                    f'VAL-LOSS: {val_loss:.4f}',
                    f'VAL-ACC: {val_acc:.4f} ',
                    f'EPOCH-TIME: {total_epoch_time//60:.0f}m{total_epoch_time%60:.0f}s',
                ])
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                pass
        end_time = time.time()
        total_time = end_time-start_time
        return best_model_wts, best_val_acc, total_time

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

    def save_model(self, model_wts, save_path):
        torch.save(model_wts, save_path)

    def test_model(self, model, dataloader, criterion):
        test_loss, test_acc = self.eval_model(model, dataloader, criterion)
        return test_loss, test_acc

    def __main__(self):
        model = self.create_model()
        dataloader = self.make_dataloader()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.num_epochs//2, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_model_wts, best_val_acc, total_time = self.train_model(
            model, 
            dataloader, 
            optimizer, 
            scheduler, 
            criterion
        )
        self.save_model(best_model_wts, self.save_path)
        model.load_state_dict(torch.load(self.save_path))
        test_loss, test_acc = self.test_model(model, dataloader['test'], criterion)
        self.logger.info(f'{"-"*10}')
        self.logger.info(
            ' '.join([
                f'TOTAL-TIME: {total_time//60:.0f}m{total_time%60:.0f}s',
                f'BEST-VAL-ACC: {best_val_acc:.4f}',
                f'TEST-LOSS: {test_loss:.4f}',
                f'TEST-ACC: {test_acc:.4f} ',
            ])
        )


if __name__ == '__main__':
    TrainOnMnist('./config/mnist.yaml')
