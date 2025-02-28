import argparse
import os
import numpy

import project_library.file_manager
import project_library.dataset_processor
from pathlib import Path

if __name__ == "__main__":
    base_directory = ""

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()



    # Context 
    ## The folder created are in Docker container so it should follow /opt/ml/processing/input

    ROOT_INPUT_FOLDER: Path = project_library.file_manager.FileInformation.get_absolute_folder_location(args.input_dir)
    ROOT_OUTPUT_FOLDER: Path = project_library.file_manager.FileInformation.get_absolute_folder_location(args.output_dir)

    fastscan_dataset_train_folder_path: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/images/train"))
    fastscan_dataset_test_folder_path: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/images/test"))
    fastscan_dataset_val_folder_path: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/images/val"))

  # 1. Retrieve the list of file name in the list for train, val and test image folder
    train_dataset_list = project_library.file_manager.FileInformation.extract_file_names_into_list(folder_path=fastscan_dataset_train_folder_path)
    test_dataset_list = project_library.file_manager.FileInformation.extract_file_names_into_list(folder_path=fastscan_dataset_test_folder_path)
    val_dataset_list = project_library.file_manager.FileInformation.extract_file_names_into_list(folder_path=fastscan_dataset_val_folder_path)
    RAW_LABEL_DATASET_FOLDER_PATH: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/raw_label_data"))



    # need to transform the list file extension to .txt from .jpg
    transformed_train_dataset_list: list = project_library.file_manager.FileInformation.update_list_containing_file_extension_into_list(data_list=train_dataset_list, extension_modification=".txt")
    transformed_test_dataset_list: list = project_library.file_manager.FileInformation.update_list_containing_file_extension_into_list(data_list=test_dataset_list, extension_modification=".txt")
    transformed_val_dataset_list: list = project_library.file_manager.FileInformation.update_list_containing_file_extension_into_list(data_list=val_dataset_list, extension_modification=".txt")
  
  # 2. Label Data Processing

      # 2.1 Create label folders for train, val and test
    label_train_folder_path: str = project_library.file_manager.CreateFolderStructure.create_folder(folder_name="train", folder_file_path="/opt/ml/processing/input/aws_datasets/fastscandataset/labels")
    label_test_folder_path: str = project_library.file_manager.CreateFolderStructure.create_folder(folder_name="test", folder_file_path="/opt/ml/processing/input/aws_datasets/fastscandataset/labels")
    label_val_folder_path: str = project_library.file_manager.CreateFolderStructure.create_folder(folder_name="val", folder_file_path="/opt/ml/processing/input/aws_datasets/fastscandataset/labels")

        # 2.1 Using the 3 different list (train, val and test) from image processing to move label data to the respective folder so that YOLO model can use the label data against the image file
    missing_train_files_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=RAW_LABEL_DATASET_FOLDER_PATH, file_name_list=transformed_train_dataset_list, destination_folder=label_train_folder_path)
    missing_test_files_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=RAW_LABEL_DATASET_FOLDER_PATH, file_name_list=transformed_test_dataset_list, destination_folder=label_test_folder_path)
    missing_validation_files_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=RAW_LABEL_DATASET_FOLDER_PATH, file_name_list=transformed_val_dataset_list, destination_folder=label_val_folder_path)


