import os
import shutil
from tqdm import tqdm
from torchvision import datasets
from concurrent.futures import ThreadPoolExecutor


def mnist_export(root: str = './data/mnist'):
    """Export MNIST data to a local folder using multi-threading.

    Args:
        root (str, optional): Path to local folder. Defaults to './data/mnist'.
    """
    for i in range(10):
        os.makedirs(os.path.join(root, f'./{i}'), exist_ok=True)
    split_list = ['train', 'test']
    data = {
        split: datasets.MNIST(
            root='./tmp',
            train=split == 'train',
            download=True
        ) for split in split_list
    }
    total = sum([len(data[split]) for split in split_list])
    with tqdm(total=total) as pbar:
        with ThreadPoolExecutor() as tp:
            for split in split_list:
                for index, (image, label) in enumerate(data[split]):
                    tmp = os.path.join(root, f'{label}/{split}_{index}.png')
                    tp.submit(image.save, tmp).add_done_callback(
                        lambda func: pbar.update()
                    )
    shutil.rmtree('./tmp')


if __name__ == '__main__':
    mnist_export('./data/mnist')
