import numpy
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing 


class CreateDataset:


    @classmethod
    def create_train_test_validation_dataset(cls, data_list: list, train_size: float, test_size:float, validation_size: float) -> tuple[list, list, list]:

        #determine the train, validation and test split ratio
        train_dataset_ratio: float = train_size

        first_pass_remaining_dataset_ratio: float = 1 - train_size

        validation_dataset_ratio: float = validation_size / first_pass_remaining_dataset_ratio

        test_dataset_ratio: float = test_size / first_pass_remaining_dataset_ratio


        # Create the train, validation and test dataset

        train_dataset, first_pass_remaining_dataset = CreateDataset.split_dataset(data_list=data_list, train_size=train_dataset_ratio , test_size=train_dataset_ratio)

        validation_dataset, test_dataset = CreateDataset.split_dataset(data_list=first_pass_remaining_dataset, train_size=validation_dataset_ratio, test_size=test_dataset_ratio)

        return train_dataset, validation_dataset, test_dataset


    @classmethod
    def split_dataset(cls, data_list: list, train_size: float, test_size: float) -> tuple[list, list]:
        
        train_list, test_list = sklearn.model_selection.train_test_split(data_list, test_size=test_size, train_size=train_size, shuffle=True)

        return train_list, test_list