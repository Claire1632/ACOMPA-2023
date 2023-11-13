import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data():
    dataset = pd.read_csv('C:/Users/ADMIN/OneDrive/Desktop/ACOMP_AFW_2023/ProcessingData/dataset/ACOMP_21_G3_0.08_36_600_6.csv')
    dataset_desc = dataset.describe(include = 'all')
    X = dataset.values[:, 0:-1].astype(float)
    y = dataset.values[:, 7].astype(float)
    # print(y)
    return X, y

