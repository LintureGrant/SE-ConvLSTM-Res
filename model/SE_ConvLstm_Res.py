import torch.nn as nn
import torch
#构建神经元
import torch.nn as nn
import torch
import numpy as np
device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias,res_rate,reduce=16,server_num = 4):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.res_rate=res_rate
        res_list=[]
        self.res_para =[] 
        
        for i in range(server_num):
            self.res_para.append(torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True).to(device))
            res_list.append(nn.Conv2d(in_channels=self.input_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=(3,3),
                                  padding=(1,1),
                                  bias=False))
        self.relu = nn.ReLU()
        self.res_list = nn.ModuleList(res_list)
        #SE module
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(4 * self.hidden_dim, 4 * self.hidden_dim // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(4 * self.hidden_dim // reduce, 4 * self.hidden_dim),
                                nn.Sigmoid())

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #print('conv_res: ',self.conv_res(input_tensor).shape,'\nconv: ',self.conv(combined).shape)
        res_split = torch.split(tensor = input_tensor,
                       split_size_or_sections = 2,
                       dim = 2)
        #print(len(res_split),res_split[0].shape)
        res_combined=torch.tensor([]).to(device)
        conv_combined=self.conv(combined)
        for i in range(len(self.res_list)):
            res_combined=torch.cat((res_combined,self.res_para[i]*self.res_list[i](res_split[i])),2)
        #print(res_combined.shape,'##',self.conv(combined).shape)
        #SE模块
        conv_b, conv_c, _, _ = conv_combined.size()
        #print(conv_combined.size())
        conv_combined_se= self.gp(conv_combined)
        #print(conv_combined_se.size())
        conv_combined_se=conv_combined_se.view(conv_b, conv_c)
        conv_combined_se = self.se(conv_combined_se).view(conv_b, conv_c, 1, 1)
        conv_combined_se = conv_combined * conv_combined_se.expand_as(conv_combined)
        #conv_combined_out = conv_combined_se + conv_combined
        combined_conv = conv_combined_se+self.res_rate*res_combined
        #combined_conv = self.conv(combined)
        #print(combined_conv.shape)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        #print(cc_f.shape,' ',c_cur.shape,'\n',cc_i.shape,' ',g.shapes)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

#构建神经网络
class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,data_row_dim=[21,21],
                 batch_first=False, bias=True, return_all_layers=False,res_rate=1):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.data_row_dim=data_row_dim
        
        #print(input_dim,' ',hidden_dim,' ',type(hidden_dim))
        self.f1 = nn.Linear(in_features =hidden_dim[0]*data_row_dim[1], out_features = data_row_dim[1])

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,res_rate=res_rate))
        
        #cell_list.append(self.f1)
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # return layer_output_list, last_state_list
        #pre=layer_output_list[0].cpu().detach().numpy()
        #print(pre.shape)
        n,l,_,_,_=layer_output_list[0].size()
        #print('1  ',layer_output_list[0].size())
        layer_output_list[0]=layer_output_list[0].view(n,l,self.data_row_dim[0],-1)
        #print('2  ',layer_output_list[0].size())
        #layer_output_list[0][0][i].squeeze(0)
        layer_output_list[0]= self.f1(layer_output_list[0])
        #print(layer_output_list[0][0][i].shape)       
        #return layer_output_list[0].squeeze(2).squeeze(0)
        #print(layer_output_list[0].shape)
        return torch.abs(layer_output_list[0].squeeze(2).squeeze(0))

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
