
import argparse
import os
import numpy

import project_library.file_manager
import project_library.image_processor
import project_library.aws_manager
import project_library.image_processor
import project_library.dataset_processor


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

    aws_dataset_folder_path: str = str(Path("/opt/ml/processing/input/aws_datasets").resolve())
    raw_dataset_folder_path: str = project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/raw_custom_dataset_images")

    #Create folder in ROOT_INPUT_FOLDER
    project_library.file_manager.CreateFolderStructure.create_folder(folder_name="aws_datasets", folder_file_path=ROOT_INPUT_FOLDER)


    fast_scan_dataset_raw_folder: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/raw"))
    fast_scan_dataset_raw_train_folder: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/raw/train"))
    fast_scan_dataset_raw_test_folder: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/raw/test"))
    fast_scan_dataset_raw_val_folder: str = str(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/raw/val"))


    # 1. Image Processing

        # 1.1 Create the dataset folders  images, label and raw with each folder containing (train, test, val) folders
    images_folder_structure, labels_folder_structure, raw_folder_structure = project_library.file_manager.CreateFolderStructure.create_dataset_folder_structure(project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets"), "fastscandataset")


        # 1.2 Extract the file name of each image in the folder for train_test_split to obtain 3 different list (train, val and test)
    raw_image_file_name_list: list = project_library.file_manager.FileInformation.extract_file_names_into_list(folder_path=raw_dataset_folder_path)

    train_dataset, validation_dataset, test_dataset = project_library.dataset_processor.CreateDataset.create_train_test_validation_dataset(data_list=raw_image_file_name_list, train_size=0.5, test_size=0.2, validation_size=0.3)


        # 1.3 After the train_test_split, using each file name from 3 different list (train, val and test) to move the images to the respective folders
    moved_train_image_file_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=raw_dataset_folder_path, file_name_list=train_dataset, destination_folder=fast_scan_dataset_raw_train_folder)
    moved_test_image_file_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=raw_dataset_folder_path, file_name_list=test_dataset, destination_folder=fast_scan_dataset_raw_test_folder)
    moved_validation_image_file_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=raw_dataset_folder_path, file_name_list=validation_dataset, destination_folder=fast_scan_dataset_raw_val_folder)


    # 2. Label Data Processing

    RAW_CUSTOM_LABEL_FOLDER_PATH: str = project_library.file_manager.FileInformation.get_absolute_folder_location("/opt/ml/processing/input/aws_datasets/fastscandataset/raw_custom_label_dataset")


        # 2.1 Using the 3 different list (train, val and test) from image processing to move label data to the respective folder so that YOLO model can use the label data against the image file
    missing_train_files_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=RAW_CUSTOM_LABEL_FOLDER_PATH, file_name_list=train_dataset, destination_folder=raw_folder_structure["train_folder_path"])
    missing_test_files_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=RAW_CUSTOM_LABEL_FOLDER_PATH, file_name_list=test_dataset, destination_folder=raw_folder_structure["test_folder_path"])
    missing_validation_files_list: list = project_library.file_manager.MoveFilesAndFolder.move_files_to_another_folder(source_folder=RAW_CUSTOM_LABEL_FOLDER_PATH, file_name_list=validation_dataset, destination_folder=raw_folder_structure["validation_folder_path"])

        # Move the image files that do not have any label file

        # 2.2 Both label data file and image file have the same file name except for the extension

