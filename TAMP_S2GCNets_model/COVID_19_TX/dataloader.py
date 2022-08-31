import torch
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon, MPG_Window_Horizon
from load_dataset import load_st_dataset, load_topo_dataset
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def triple_data_loader(X, MPG, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, MPG, Y = TensorFloat(X), TensorFloat(MPG), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, MPG, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_topo_dataloader(args, H_type, normalizer = 'std', single=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #spilit dataset by days or by ratio

    x, y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    print('x:', x.shape)
    print('y:', y.shape)

    _topo_data_ = load_topo_dataset(H_type)
    topo_data = MPG_Window_Horizon(_topo_data_, args.lag, args.horizon, single)
    print('MPG:', topo_data.shape)

    print("================+++++++++++++================")
    print('This function is designed for COVID networks')
    print("================+++++++++++++================")

    # only for train and test
    single = False
    fraction = 0.8
    x_tra, y_tra = x[0:int(np.round(fraction * x.shape[0])),...], y[0:int(np.round(fraction * x.shape[0])),...]
    topo_tra = topo_data[0:int(np.round(fraction * x.shape[0]))]
    x_test, y_test = x[int(np.round(fraction * x.shape[0])):,...], y[int(np.round(fraction * x.shape[0])):,...]
    topo_test = topo_data[int(np.round(fraction * x.shape[0])):]
    print('Train: ', x_tra.shape, topo_tra.shape, y_tra.shape)
    print('Test: ', x_test.shape, topo_test.shape, y_test.shape)
    ##############get triple dataloader######################
    train_dataloader = triple_data_loader(x_tra, topo_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = triple_data_loader(x_test, topo_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, None, test_dataloader, scaler # None is for validation
