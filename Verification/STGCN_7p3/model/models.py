import torch
import torch.nn as nn

from model import layers

class STGCN_ChebConv(nn.Module):
    # STGCN(ChebConv) contains 'TGTND TGTND TNFF' structure
    # ChebConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
    # It is an Autoregressive(AR) filter in Finite Impulse Response(FIR) filters.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, class_mat, use_class, n_pred, drop_rate):
        super(STGCN_ChebConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], gated_act_func, graph_conv_type, chebconv_matrix, drop_rate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        # print("Ko: ", self.Ko)
        self.class_mat = class_mat
        self.use_class = use_class
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, gated_act_func, class_mat, use_class, n_pred, drop_rate)
        elif self.Ko == 0:
            if self.use_class:
                self.fc1 = nn.Linear(blocks[-3][-1] + self.class_mat.shape[0], blocks[-2][0])
            else:
                self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])
            self.act_func = 'sigmoid'
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.elu = nn.ELU()
            self.do = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x_stbs = self.st_blocks(x)
        if self.Ko > 1:
            x_out = self.output(x_stbs)
        elif self.Ko == 0:
            # print("x_stbs.shape: ", x_stbs.shape)
            if self.use_class:
                x_fc1 = self.fc1(torch.cat((x_stbs.permute(0, 2, 3, 1), torch.repeat_interleave(torch.repeat_interleave(self.class_mat.permute(1,0).unsqueeze(dim=0).unsqueeze(dim=0), repeats = x_stbs.shape[0], dim=0), repeats = x_stbs.shape[2], dim = 1)), -1))
            else:
                x_fc1 = self.fc1(x_stbs.permute(0, 2, 3, 1))
            # print("x_fc1.shape: ", x_fc1.shape)
            if self.act_func == 'sigmoid':
                x_act_func = self.sigmoid(x_fc1)
            elif self.act_func == 'tanh':
                x_act_func = self.tanh(x_fc1)
            elif self.act_func == 'relu':
                x_act_func = self.relu(x_fc1)
            elif self.act_func == 'leaky_relu':
                x_act_func = self.leaky_relu(x_fc1)
            elif self.act_func == 'elu':
                x_act_func = self.elu(x_fc1)
            # print("x_act_func.shape: ", x_act_func.shape)
            x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)
            # print("x_fc2.shape: ", x_fc2.shape)
            x_out = x_fc2
            # print("x_out.shape: ", x_out.shape)
        return x_out

class STGCN_GCNConv(nn.Module):
    # STGCN(GCNConv) contains 'TGTND TGTND TNFF' structure
    # GCNConv is the graph convolution from GCN.
    # GCNConv is not the first-order ChebConv, because the renormalization trick is used.
    # It is an Autoregressive(AR) filter in Finite Impulse Response(FIR) filters.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GCNConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Kt, Ks, blocks, T, n_vertex, gated_act_func, graph_conv_type, gcnconv_matrix, class_mat, use_class, n_pred, drop_rate):
        super(STGCN_GCNConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(Kt, Ks, n_vertex, blocks[l][-1], blocks[l+1], gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        # print("Ko: ", self.Ko)
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, gated_act_func, class_mat, use_class, n_pred, drop_rate)
        elif self.Ko == 0:
            if self.use_class:
                self.fc1 = nn.Linear(blocks[-3][-1] + self.class_mat.shape[0], blocks[-2][0])
            else:
                self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0])
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0])
            self.act_func = 'sigmoid'
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.elu = nn.ELU()
            self.do = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x_stbs = self.st_blocks(x)
        if self.Ko > 1:
            x_out = self.output(x_stbs)
        elif self.Ko == 0:
            # print("x_stbs.shape: ", x_stbs.shape)
            if self.use_class:
                x_fc1 = self.fc1(torch.cat((x_stbs.permute(0, 2, 3, 1), torch.repeat_interleave(torch.repeat_interleave(self.class_mat.permute(1,0).unsqueeze(dim=0).unsqueeze(dim=0), repeats = x_stbs.shape[0], dim=0), repeats = x_stbs.shape[2], dim = 1)), -1))
            else:
                x_fc1 = self.fc1(x_stbs.permute(0, 2, 3, 1))
            # print("x_fc1.shape: ", x_fc1.shape)
            if self.act_func == 'sigmoid':
                x_act_func = self.sigmoid(x_fc1)
            elif self.act_func == 'tanh':
                x_act_func = self.tanh(x_fc1)
            elif self.act_func == 'relu':
                x_act_func = self.relu(x_fc1)
            elif self.act_func == 'leaky_relu':
                x_act_func = self.leaky_relu(x_fc1)
            elif self.act_func == 'elu':
                x_act_func = self.elu(x_fc1)
            # print("x_act_func.shape: ", x_act_func.shape)
            x_fc2 = self.fc2(x_act_func).permute(0, 3, 1, 2)
            # print("x_fc2.shape: ", x_fc2.shape)
            x_out = x_fc2
            # print("x_out.shape: ", x_out.shape)
        
        return x_out