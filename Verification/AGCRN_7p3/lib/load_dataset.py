import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset in ['GLOBAL4', 'ATLANTIC1', 'PACIFIC1', 'SOUTH_ATLANTIC1']:
        data_path = os.path.join('../data/{}/{}.npz'.format(dataset, dataset))
        data = np.load(data_path)['data']
        classinfo = np.load(data_path)['classinfo']
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    if len(classinfo.shape) == 2:
        classinfo = np.expand_dims(classinfo, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    print('The type of the data is: ', type(data))
    print('Load %s Classinfo shaped: ' % dataset, classinfo.shape)
    print('The type of the classinfo is: ', type(classinfo))
    return data, classinfo
