import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
import tarfile
import pickle
import requests
import gzip

class BaseModel:

    def load_pickle_from_tar(tar_path, pickle_path):
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmember(pickle_path)
            f = tar.extractfile(member)
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict

    def download_cifar10():
        working_path = os.getcwd()
        cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        file_name_gz = "cifar-10-python.tar.gz"
        file_name_tar = "cifar-10-python.tar"
  
        file_path_gz = os.path.join(working_path, file_name_gz)
        file_path_tar = os.path.join(working_path, file_name_tar)

        if os.path.exists(file_path_tar):
            print("The file cifar-10-python.tar already exists")
        else:
            print("Downloading CIFAR-10 data...")
            response = requests.get(cifar_url", stream=True)
            response.raise_for_status()
            with open(file_path_gz, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            with gzip.open(file_path_gz, 'rb') as f_in:
                with open(file_path_tar, 'wb') as f_out:
                    for chunk in f_in:
                        f_out.write(chunk)
            print
        return file_path_tar
    
    def __init__(self):
        pass

    def load_data():
        """
        Load the model from the specified path.
        """
        tar_path = download_cifar10()  # File in the same working directory

        # Load training data
        train_data = []
        train_labels = []

        print("Loading training data...")
        # The path structure below is based on the standard CIFAR-10 distribution
        for i in range(1, 6):
            batch_path = f'cifar-10-batches-py/data_batch_{i}'
            print(f"Loading batch {i}...")
            batch_dict = load_pickle_from_tar(tar_path, batch_path)
            train_data.append(batch_dict[b'data'])
            train_labels.extend(batch_dict[b'labels'])

        train_data = np.vstack(train_data)

        # Each image is 32x32 pixels with 3 color channels (RGB)
        X_train = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, 32, 32, 3)
        X_test = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, 32, 32, 3)
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Print updated shapes
        print("Training data shape after reshaping:", X_train.shape)
        print("Test data shape after reshaping:", X_test.shape)
        pass
    
    def preprocess(self, data):
        """
        Preprocess the input data.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
        pass
