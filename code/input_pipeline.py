from torch.utils.data import Dataset
import pandas as pd
import pickle
import torch
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from random import sample


class Dataset(Dataset):

    def __init__(self, data_file_training, data_file_testing, post_map_path, feature_path, aux_path, attr_path, min_images, training=True):

        self.training = training
        self.input_points_training = pd.read_csv(data_file_training, header=None, dtype='str')
        self.input_points_testing = pd.read_csv(data_file_testing, header=None, dtype='str')

        with open(post_map_path, 'r') as f:
            self.code_list = pickle.load(f)

        with open(attr_path, 'r') as f:
            attrs = [el.replace('\n', '').split(',') for el in f.readlines()]

        self.image_feature = np.load(feature_path)

        aux_data = pd.read_csv(aux_path, index_col=0)
        self.aux_list = [row['Aux_id'] for index, row in aux_data.iterrows() if row['Aux_id'] == row['Aux_id']]
        self.input_points_training = {el: self.input_points_training.loc[self.input_points_training.iloc[:, 0] == el].values[:,1].tolist() for el in self.aux_list}
        self.input_points_testing = {el: self.input_points_testing.loc[self.input_points_testing.iloc[:, 0] == el].values[:,1].tolist() for el in self.aux_list}
        self.aux_list = [el for el in self.aux_list if len(self.input_points_training[el]) > min_images] #to cut entities with less than min_images images
        aux_numeric_data = aux_data.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
        aux_selection = aux_numeric_data.values
        self.aux_headers = list(aux_numeric_data.columns.values)

        mm_scaler = MinMaxScaler()
        aux_selection = mm_scaler.fit_transform(aux_selection)

        self.aux_data = {row['Aux_id']: aux_selection[index, :] for index, row in aux_data.iterrows() if row['Aux_id'] is not np.nan}

        self.attr_names = zip(*attrs)[0]
        self.attr_codes = zip(*attrs)[1]
        self.attr_inds = [self.aux_headers.index(attr) for attr in zip(*attrs)[1]]
        self.min_images = min_images

    def __len__(self):
        return len(self.aux_list)

    def __getitem__(self, idx):

        if self.training:
            aux_1_name = self.aux_list[idx]
            col = random.sample(self.input_points_training[aux_1_name], self.min_images)
            images_1 = self.image_feature[[self.code_list[el] for el in col], :]

            aux_2 = sample([x for x in range(len(self.aux_list)) if x != idx], 1)[0]
            aux_2_name = self.aux_list[aux_2]
            col = random.sample(self.input_points_training[aux_2_name], self.min_images)
            images_2 = self.image_feature[[self.code_list[el] for el in col], :]

        else:
            aux_1_name = self.aux_list[idx]
            col = random.sample(self.input_points_testing[aux_1_name], self.min_images)
            images_1 = self.image_feature[[self.code_list[el] for el in col], :]

            aux_2 = sample([x for x in range(len(self.aux_list)) if x != idx], 1)[0]
            aux_2_name = self.aux_list[aux_2]
            col = random.sample(self.input_points_testing[aux_2_name], self.min_images)
            images_2 = self.image_feature[[self.code_list[el] for el in col], :]

        _sample = {
                  'image_1': torch.from_numpy(images_1),
                  'image_2': torch.from_numpy(images_2),
                  'aux_1': idx,
                  'aux_2': aux_2,
                  'label_1': np.array(self.aux_data[aux_1_name][np.array(self.attr_inds)]),
                  'label_2': np.array(self.aux_data[aux_2_name][np.array(self.attr_inds)])
                  }

        return _sample

