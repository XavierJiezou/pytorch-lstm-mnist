'''
YAML config file is recommended.
'''
import argparse

class OptionSummary():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.__main__()
        self.parser = self.parser.parse_args()
    
    def data_option(self):
        self.parser.add_argument('-dr', '--data_root', default='./data/mnist', type=str, required=True, help='Path to data')
        self.parser.add_argument('-bs', '--batch_size', default=64, type=int, required=False, help='How many samples per batch to load')
    
    def model_option(self):
        self.parser.add_argument('-is', '--input_size', default=28, type=int, required=True, help='Number of expected features in the input')
        self.parser.add_argument('-hs', '--hidden_size', default=64, type=int, required=False, help='Number of features in the hidden state')
        self.parser.add_argument('-nl', '--num_layers', default=1, type=int, required=False, help='Number of recurrent layers')
        self.parser.add_argument('-os', '--output_size', default=10, type=int, required=True, help='Number of expected features in the output')

    def train_option(self):
        self.parser.add_argument('-ne', '--num_epochs', default=10, type=int, required=False, help='How many epochs to use for data training')
        self.parser.add_argument('-sl', '--sequence_length', default=28, type=int, required=True, help='Length of the input sequence')
        self.parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, required=False, help='Learning_rate')
        self.parser.add_argument('-d', '--device', default='cpu', type=str, required=False, help='On which a `torch.Tensor` is or will be allocated')
        self.parser.add_argument('-gi', '--gpu_id', default='0', type=str, required=False, help='Which gpu to be selected to train the model')
        self.parser.add_argument('-sp', '--save_path', default='model.pth', type=str, required=False, help='Path to save the trained model')

    def __main__(self):
        self.data_option()
        self.model_option()
        self.train_option()


if __name__ == '__main__':
    print(OptionSummary().parser)
'''bash
$ python .\script\option_summary.py -dr './data/mnist' -is 28 -os 10 -sl 28
Namespace(data_root='./data/mnist', batch_size=64, input_size=28, hidden_size=64, num_layers=1, output_size=10, num_epochs=10, sequence_length=28, learning_rate=0.001, device='cpu', gpu_id='0', save_path='model.pth')
'''
