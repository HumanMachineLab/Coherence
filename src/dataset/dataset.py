import sys
import os
import pandas
import json
import config

sys.path.append("../../")

default_data_path = os.path.join("..", "data")


class RawData:
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.split = None

    def get_data(self, split="train"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/{self.dataset_type}/wikisection_en_{self.dataset_type}_{split}.json"
        data = None
        with open(data_path) as f:
            data = json.load(f)

        assert data is not None, "data could not be imported"

        return data
    
    def get_qmsum_data(self, split="train", dataset="Academic"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/qmsum/{dataset}/{split}"
        data = []

        for filename in os.listdir(data_path):
            f = os.path.join(data_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f) as f:
                    temp_data = json.load(f)
                    assert data is not None, f"data: {filename} could not be imported"
                    data.append(temp_data)
        
        assert len(data) > 0, "data could not be imported"

        return data
    
    def get_superseg_data(self, split="train"):
        self.split = split
        data_path = f"{config.root_path}/raw_data/superseg/segmentation_file_{split}.json"
        data = None
        with open(data_path) as f:
            data = json.load(f)

        assert data is not None, "data could not be imported"

        return data
    
    
    def get_wiki_data(self):
        data_path = f"{config.root_path}/raw_data/wiki727k"
        data = []

        for filename in os.listdir(data_path):
            f = os.path.join(data_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                with open(f, mode="r", encoding="utf-8") as f:
                    for line in f:
                        data.append(line)
        
        assert len(data) > 0, "data could not be imported"

        return data


class DatasetMixin:
    # save dataset and retrieve dataset information.
    # load datasets based on parameters.
    @staticmethod
    def get_datasets():
        df = pandas.read_csv(os.path.join(default_data_path, "datasets.csv"))
        print(df)

    @staticmethod
    def save_dataset():
        pass
