import torch
import numpy as np
import torch.utils.data
import os

def load_st_dataset(dataset):
    #output B, N, D
    if dataset in ['NORTH_ATLANTIC_025', 'NORTH_PACIFIC_025', 'SOUTH_ATLANTIC_025', 'SOUTH_ATLANTIC1']:
        data_path = os.path.join('./data/{}.npz'.format(dataset, dataset))
        data = np.load(data_path)['data']
    else:
        raise ValueError

    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    print('The type of the data is: ', type(data))
    return data

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    print('data_len:',data_len)
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def Add_Window_Horizon(data, window=7, horizon=3, step = 1):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    print(index, end_index, length)
    while index < end_index:
        X.append(np.swapaxes(data[index:index+window], 0, 1))
        index = index + step
    X = np.array(X)
    Y = np.copy(X)
    return X, Y

def data_loader(device, X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() and device != 'cpu' else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def get_dataloader(args, normalizer = 'std'):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # T * N
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio

    data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)

    x_tra, y_tra = Add_Window_Horizon(data_train, args.window, args.horizon, 28)
    print('Train: ', x_tra.shape, y_tra.shape)
    train_dataloader = data_loader(args.device, x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    del x_tra
    del y_tra

    x_val, y_val = Add_Window_Horizon(data_val, args.window, args.horizon, 28)
    print('Val: ', x_val.shape, y_val.shape)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(args.device, x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    del x_val
    del y_val

    x_test, y_test = Add_Window_Horizon(data_test, args.window, args.horizon, 28)
    print('Test: ', x_test.shape, y_test.shape)
    test_dataloader = data_loader(args.device, x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    del x_test
    del y_test
    
    ##############get dataloader######################
   
    
    return train_dataloader, val_dataloader, test_dataloader, scaler

if __name__ == '__main__':
    import argparse
    DATASET = 'GLOBAL3'
    NODE_NUM = 5389
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()
    train_dataloader, scaler = get_dataloader(args, normalizer = 'std')
