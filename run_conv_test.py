import numpy as np
from get_data import *
from dataset import traffic_dataset
import torch.utils.data.dataloader as DataLoader
from model import ConvLstm
import torch
import time



def load_data():
    path = './data/timeSpace_dataset.txt'
    data = get_timeSpaceData(path)
    # normalization
    data = np.array(data) / (np.array(data).max() - np.array(data).min())
    print(np.array(data).shape)
    return data


def sliding_windows(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# def loss_function(y_true, y_predict):
#     assert y_true.shape == y_predict.shape
#     a,h,w= y_true.shape
#     loss = torch.tensor(0.0).to(device)
#     std_loss= torch.tensor(0.0).to(device)
#     for i in range(len(y_true)):
#         users_loss=torch.tensor([(y_true[i][k][n] - y_predict[i][k][n]) ** 2 for k in range(len(y_true[0])) for n in range(len(y_true[0][0]))])
#         loss += torch.sum((y_true[i] - y_predict[i]) ** 2)
#         std_loss+=users_loss.std()
#     return (loss+std_loss)/a

def loss_function(y_predict , y_true ):
    assert y_true.shape == y_predict.shape
    a,h,w= y_true.shape
    loss = torch.tensor(0.0).to(device)
    for i in range(len(y_true)):
        loss += torch.sum((y_true[i] - y_predict[i]) ** 2)
    return loss/a


if __name__ == "__main__":

    # -----parameter------------------------
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = ConvLstm.device
    train_ratio = 0.6
    batch_size = 100
    input_dim = 10
    hidden_dim = 32
    kernel_size = (5,5)
    num_layers = 1
    lr = 1e-3
    gamma = 0.97
    epoch = 40


    # -----dataset------------------------
    data = load_data()
    train_X, train_Y = sliding_windows(data[:int(0.6 * len(data))])
    test_X, test_Y = sliding_windows(data[int(0.6 * len(data)):])
    train_set = traffic_dataset(train_X, train_Y)
    train_data = DataLoader.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_set = traffic_dataset(test_X, test_Y)
    test_data = DataLoader.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # -----model------------------------
    model = ConvLstm.ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers, data_row_dim=[len(data[0]), len(data[0][0])],
                                               batch_first=True, res_rate=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # -----train------------------------
    print('start:' + time.asctime(time.localtime(time.time())))
    # loss_function = nn.MSELoss().to(device)
    losses = []
    losses_test = []
    for i in range(40):
        model.train()
        loss_batch = 0
        loss_train_batch = 0
        loss_test_batch = 0
        for j, item in enumerate(train_data):
            train_x, train_y = item
            out = model(train_x.unsqueeze(0))
            # print(out.shape,train_y.shape)
            loss = loss_function(out, train_y)
            loss_batch += loss.item()
            loss_train_batch += loss.item()
            torch.autograd.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if (j + 1) % 100 == 0:
                print('Epoch: {},batch:{}, Loss:{:.5f}'.format(i + 1, j + 1, loss_batch / 50))  # loss_test:{:.5f}
                loss_batch = 0
        losses.append(loss_train_batch / len(train_data))
        scheduler.step()
        with torch.no_grad():
            model.eval()
            for n, item in enumerate(test_data):
                test_x, test_y = item
                out_test = model(test_x.unsqueeze(0))
                loss_test = loss_function(out_test, test_y)
                loss_test_batch += loss_test.item()
                # record validation loss
                # print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
            losses_test.append(loss_test_batch / len(test_data))
        print('Epoch: {},loss_train:{:.5f},loss_test:{:.5f}'.format(i + 1, loss_train_batch / len(train_data),
                                                                    loss_test_batch / len(test_data)) + time.asctime(
            time.localtime(time.time())))  #
    print('end:' + time.asctime(time.localtime(time.time())))