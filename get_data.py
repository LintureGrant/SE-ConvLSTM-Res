#file_path = 'DataSet.txt'
import numpy as np

def get_data(file_path):
    data_list = []
    with open(file_path, 'rt', encoding='UTF-8') as f:
        data = f.read()

    data_temp_list = data.split('\n')[:-1]
    for v in data_temp_list:
        data_temp_list_i = v.split('\t')
        for i in range(len(data_temp_list_i)):
            data_temp_list_i[i] = float(data_temp_list_i[i])
        data_list.append(data_temp_list_i)

    return data_list


def get_timeSpaceData(file_path):
    timeSpaceData=[]
    with open(file_path, 'rt', encoding='UTF-8') as f:
        data = f.read()
    #去掉最后一个换行符,写成#标识
    data_temp_list = data.split('\n')
    data_temp_list[-1]='#'
    #timeSpaceData = np.zero((len(data_temp_list),21,21))
    #初始化某一时刻的数据即21*21的矩阵
    onetimeData = []
    for i in range(len(data_temp_list)):
        data_temp_list_i = data_temp_list[i].split('\t')
        #当读完一个时刻的数据后，数据加入timeSpaceData，并使当前onetimeData置于0
        if data_temp_list_i[0]=='#' :
            if onetimeData!=[]:
                timeSpaceData.append(onetimeData)
            onetimeData = []
            continue
        for j in range(len(data_temp_list_i)):
            #print('##',data_temp_list_i[j],'##')
            data_temp_list_i[j].replace(' ','')
            if data_temp_list_i[j]!='':
                data_temp_list_i[j] = float(data_temp_list_i[j])
        if data_temp_list_i!=[]:
            onetimeData.append(data_temp_list_i)
    return timeSpaceData