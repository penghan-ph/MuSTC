import logging
import os
import argparse
import configparser
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from torchsummary import summary

from script import dataloader, utility, earlystopping
from model import models

#import nni

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    set_seed(worker_id)

def classinfo_cuda(class_info):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    class_info = TensorFloat(class_info)
    return class_info

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN for SST prediction')
    parser.add_argument('--enable_cuda', type=bool, default='True',
                        help='enable CUDA, default as True')
    # parser.add_argument('--n_pred', type=int, default=3, 
    #                     help='the number of time interval for predcition, default as 3')
    parser.add_argument('--epochs', type=int, default=500,
                        help='epochs, default as 500')
    parser.add_argument('--dataset_config_path', type=str, default='./config/data/south_atlantic1_class.ini',
                        help='the path of dataset config file, global4.ini for global4, \
                            pacific2.ini for pacific2, and atlantic1.ini for atlantic1')
    parser.add_argument('--model_config_path', type=str, default='./config/model/chebconv_sym_glu.ini',
                        help='the path of model config file, chebconv_sym_glu.ini for STGCN(ChebConv, Ks=3, Kt=3), \
                            and gcnconv_sym_glu.ini for STGCN(GCNConv, Kt=3)')
    parser.add_argument('--opt', type=str, default='AdamW',
                        help='optimizer, default as AdamW')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    config = configparser.ConfigParser()

    def ConfigSectionMap(section):
        dict1 = {}
        options = config.options(section)
        for option in options:
            try:
                dict1[option] = config.get(section, option)
                if dict1[option] == -1:
                    logging.debug("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

    model_config_path = args.model_config_path
    dataset_config_path = args.dataset_config_path

    config.read(dataset_config_path, encoding="utf-8")

    dataset = ConfigSectionMap('data')['dataset']
    n_his = int(ConfigSectionMap('data')['n_his'])
    n_pred = int(ConfigSectionMap('data')['n_pred'])
    Kt = int(ConfigSectionMap('data')['kt'])
    stblock_num = int(ConfigSectionMap('data')['stblock_num'])
    if ((Kt - 1) * 2 * stblock_num > n_his) or ((Kt - 1) * 2 * stblock_num <= 0):
        raise ValueError(f'ERROR: {Kt} and {stblock_num} are unacceptable.')
    Ko = n_his - (Kt - 1) * 2 * stblock_num
    drop_rate = float(ConfigSectionMap('data')['drop_rate'])
    batch_size = int(ConfigSectionMap('data')['batch_size'])
    learning_rate = float(ConfigSectionMap('data')['learning_rate'])
    weight_decay_rate = float(ConfigSectionMap('data')['weight_decay_rate'])
    step_size = int(ConfigSectionMap('data')['step_size'])
    gamma = float(ConfigSectionMap('data')['gamma'])
    use_class = bool(int(ConfigSectionMap('data')['use_class']))
    data_path = ConfigSectionMap('data')['data_path']
    wam_path = ConfigSectionMap('data')['wam_path']
    class_path = ConfigSectionMap('data')['class_path']
    model_save_path = ConfigSectionMap('data')['model_save_path']

    config.read(model_config_path, encoding="utf-8")

    gated_act_func = ConfigSectionMap('casualconv')['gated_act_func']
    
    graph_conv_type = ConfigSectionMap('graphconv')['graph_conv_type']
    if (graph_conv_type != "chebconv") and (graph_conv_type != "gcnconv"):
        raise NotImplementedError(f'ERROR: {graph_conv_type} is not implemented.')
    else:
        graph_conv_type = graph_conv_type
    
    Ks = int(ConfigSectionMap('graphconv')['ks'])
    if (graph_conv_type == 'gcnconv') and (Ks != 2):
        Ks = 2
    
    mat_type = ConfigSectionMap('graphconv')['mat_type']

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])
    
    time_pred_str = str(n_pred) + '_days'
    model_name = ConfigSectionMap('graphconv')['model_name']
    model_save_path = model_save_path + model_name + '_' + dataset + '_' + time_pred_str + '.pth'

    adj_mat = dataloader.load_weighted_adjacency_matrix(wam_path)
    class_mat = classinfo_cuda(dataloader.load_classinfo_adjacency_matrix(class_path))

    print("use_class: ", use_class)

    n_vertex_vel = pd.read_csv(data_path, header=None).shape[1]
    n_vertex_adj = pd.read_csv(wam_path, header=None).shape[1]
    if n_vertex_vel != n_vertex_adj:
        raise ValueError(f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
    else:
        n_vertex = n_vertex_vel

    opt = args.opt
    epochs = args.epochs

    if graph_conv_type == "chebconv":
        if (mat_type != "wid_sym_normd_lap_mat") and (mat_type != "wid_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {args.mat_type} is wrong.')
        mat = utility.calculate_laplacian_matrix(adj_mat, mat_type)
        chebconv_matrix = torch.from_numpy(mat).float().to(device)
        stgcn_chebconv = models.STGCN_ChebConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, class_mat, use_class, n_pred, drop_rate).to(device)
        model = stgcn_chebconv

    elif graph_conv_type == "gcnconv":
        if (mat_type != "hat_sym_normd_lap_mat") and (mat_type != "hat_rw_normd_lap_mat"):
            raise ValueError(f'ERROR: {args.mat_type} is wrong.')
        mat = utility.calculate_laplacian_matrix(adj_mat, mat_type)
        gcnconv_matrix = torch.from_numpy(mat).float().to(device)
        stgcn_gcnconv = models.STGCN_GCNConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, gcnconv_matrix, class_mat, use_class, n_pred, drop_rate).to(device)
        model = stgcn_gcnconv

    return device, n_his, n_pred, model_save_path, data_path, n_vertex, batch_size, drop_rate, opt, epochs, graph_conv_type, model, learning_rate, weight_decay_rate, step_size, gamma

def data_preparate(data_path, device, n_his, n_pred, batch_size):
    data_col = pd.read_csv(data_path, header=None).shape[0]
    # recommended dataset split rate as train: val: test = 60: 20: 20, 70: 15: 15 or 80: 10: 10
    # using dataset split rate as train: val: test = 60: 20: 20
    val_and_test_rate = 0.2

    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val, test = dataloader.load_data(data_path, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)

    x_train, y_train = dataloader.data_transform(train, n_his, n_pred, device)
    x_val, y_val = dataloader.data_transform(val, n_his, n_pred, device)
    x_test, y_test = dataloader.data_transform(test, n_his, n_pred, device)

    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return zscore, train_iter, val_iter, test_iter

def main(learning_rate, weight_decay_rate, graph_conv_type, model_save_path, model, n_his, n_vertex, step_size, gamma, opt):
    loss = nn.MSELoss()
    learning_rate = learning_rate
    weight_decay_rate = weight_decay_rate
    early_stopping = earlystopping.EarlyStopping(patience=60, path=model_save_path, verbose=True)

    model_stats = summary(model, (1, n_his, n_vertex))

    if opt == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    elif opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
    else:
        raise ValueError(f'ERROR: optimizer {opt} is undefined.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return loss, early_stopping, optimizer, scheduler

def train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter):
    min_val_loss = np.inf
    for epoch in range(epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            # print("model(x).shape: ", model(x).shape)
            # print("y.shape: ", y.shape)
            y_pred = torch.reshape(model(x), (model(x).shape[0], model(x).shape[1], model(x).shape[3]))  # [batch_size, n_pred, num_nodes]
            # print("y_pred.shape: ", y_pred.shape)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        val_loss = val(model, val_iter)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        early_stopping(val_loss, model)
        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'.\
            format(epoch+1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if early_stopping.early_stop:
            print("Early stopping.")
            break
    print('\nTraining finished.\n')

def val(model, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_iter:
            # print("model(x).shape: ", model(x).shape)
            # print("y.shape: ", y.shape)
            y_pred = torch.reshape(model(x), (model(x).shape[0], model(x).shape[1], model(x).shape[3]))  # [batch_size, n_pred, num_nodes]
            # print("y_pred.shape: ", y_pred.shape)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n
    
def test(zscore, loss, model, test_iter, n_pred):
    best_model = model
    best_model.load_state_dict(torch.load(model_save_path))
    test_MSE = utility.evaluate_model(best_model, loss, test_iter)
    print('Test loss {:.6f}'.format(test_MSE))
    for i in range(n_pred):
        test_MAE, test_RMSE = utility.evaluate_metric_index(best_model, test_iter, zscore, i)
        print(f'Horizon {i:02d}, MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f}')
    test_MAE, test_RMSE = utility.evaluate_metric(best_model, test_iter, zscore)
    print(f'Average Horizon, MAE {test_MAE:.6f} | RMSE {test_RMSE:.6f}')

if __name__ == "__main__":
    # For stable experiment results
    SEED = 1608825600
    set_seed(SEED)

    # For multi-threading dataloader
    #worker_init_fn(SEED)

    # Logging
    #logger = logging.getLogger('stgcn')
    #logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)

    device, n_his, n_pred, model_save_path, data_path, n_vertex, batch_size, drop_rate, opt, epochs, graph_conv_type, model, learning_rate, weight_decay_rate, step_size, gamma = get_parameters()
    zscore, train_iter, val_iter, test_iter = data_preparate(data_path, device, n_his, n_pred, batch_size)
    loss, early_stopping, optimizer, scheduler = main(learning_rate, weight_decay_rate, graph_conv_type, model_save_path, model, n_his, n_vertex, step_size, gamma, opt)

    # Training
    train(loss, epochs, optimizer, scheduler, early_stopping, model, model_save_path, train_iter, val_iter)

    # Testing
    test(zscore, loss, model, test_iter, n_pred)
