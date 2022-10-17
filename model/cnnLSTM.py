from torch import nn
import torch
#CNN-LSTM模型构建
# 定义网络结构
class CNN_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size,r_size=8,c_size=60, output_size=1, num_layers=1,conv_outsize=10):
        super(CNN_LSTM,self).__init__()
        self.input_size=input_size
        self.conv_outsize=conv_outsize
        self.r_size=r_size
        self.c_size=c_size
        #输入(1,10,8,60),输出(1, 36, 8, 60)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10,
                            out_channels=20,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            #             torch.nn.BatchNorm2d(20),
            #             torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(20,conv_outsize,3,1,1),
        )
        sequence_size=self.input_size*r_size*c_size
        self.lstm = nn.LSTM(sequence_size, hidden_size*r_size*c_size, num_layers)
        self.fc = nn.Linear(hidden_size*r_size*c_size,output_size*r_size*c_size)
    def forward(self, x):
        x = x.squeeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.lstm(x.view(-1,1,self.r_size*self.c_size*self.conv_outsize))
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view( -1,self.r_size,self.c_size)  # 把形状改回来
        return x