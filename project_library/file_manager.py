import zipfile
import os
import numpy
import cv2
from pathlib import Path
import shutil
import project_library.dataset_processor



class FileInformation:
    
    @classmethod
    def get_absolute_folder_location(cls, folder_name: str) -> Path:
        current_relative_folder_path: Path = Path(folder_name)
        
        return current_relative_folder_path.resolve()
    
    @classmethod
    def extract_file_names_into_list(cls, folder_path: Path) -> list[str]:

        file_name_list: list[str] = []

        for file_name in os.listdir(folder_path):
            file_name_list.append(file_name)

        return file_name_list


        

class FileExtraction:

    @staticmethod    
    def extract_zip_archive_file(self, zip_archive_source_file_path: str, destination_extraction_file_path: str) -> None:
        
        with zipfile.ZipFile(file=zip_archive_source_file_path, mode="r") as zip_ref:
            zip_ref.extractall(path=destination_extraction_file_path)

        print(f"The extracted files from the zip archive are located in {destination_extraction_file_path} ")



class MoveFilesAndFolder:

    @classmethod
    def move_images_to_train_validation_test_folder(cls, source_folder: Path, file_name_list: list[str], destination_folder: Path, dataset_identifier: dict[str]) -> tuple[str, str, str]:

        if not all(identifier in dataset_identifier for identifier in ["TRAIN", "VALIDATION", "TEST"]):
            raise Exception(f"Please make sure dataset_identifier have the following keys: {["TRAIN", "VALIDATION", "TEST"]}")



        train_dataset_folder_path: str = os.path.join(destination_folder.resolve(), dataset_identifier["TRAIN"]) 
        validation_dataset_folder_path: str = os.path.join(destination_folder.resolve(), dataset_identifier["VALIDATION"])
        test_dataset_folder_path: str = os.path.join(destination_folder.resolve(), dataset_identifier["TEST"])

        # Split the data 
        train_dataset, validation_dataset, test_dataset = project_library.dataset_processor.CreateDataset.create_train_test_validation_dataset(file_name_list, train_size=0.5, validation_size=0.3, test_size=0.2)

        MoveFilesAndFolder.move_files_to_another_folder(source_folder=source_folder, file_name_list=train_dataset, destination_folder=train_dataset_folder_path)
        MoveFilesAndFolder.move_files_to_another_folder(source_folder=source_folder, file_name_list=validation_dataset, destination_folder=validation_dataset_folder_path)
        MoveFilesAndFolder.move_files_to_another_folder(source_folder=source_folder, file_name_list=test_dataset, destination_folder=test_dataset_folder_path)

        return train_dataset_folder_path, validation_dataset_folder_path, test_dataset_folder_path
        

    @classmethod
    def move_files_to_another_folder(cls, source_folder: Path, file_name_list: list[str], destination_folder: Path) -> None:
        
        for file_name in file_name_list:

            shutil.copyfile(src=os.path.join(source_folder, file_name), dst=os.path.join(destination_folder, file_name), follow_symlinks=False)

        return None

#need to create sub folder "mydata" to contain all images that are captured from the 3 minute recorded video

# The subfolder "mydata" should contain the same structure in "crack400" folder

# Process the labelled images and labels 

class CreateFolderStructure:

    IMAGES: str = "images"
    LABELS: str = "labels"
    RAW: str = "raw"

    TRAIN: str = "train"
    TEST: str = "test"
    VALIDATION: str = "val"

    FOLDER_TYPE: dict[str] = {
        TRAIN: TRAIN,
        TEST: TEST,
        VALIDATION: VALIDATION
    }

    @classmethod
    def replicate_crack400_folder_structure(cls, root_folder_file_path: str, dataset_folder_name: str) -> tuple[dict, dict, dict]:

        train_folder_structure, test_folder_structure, validation_folder_structure = CreateFolderStructure.create_new_dataset_folder(root_folder_file_path=root_folder_file_path, dataset_folder_name=dataset_folder_name)

        raw_folder_path: str = os.path.join(root_folder_file_path, dataset_folder_name, CreateFolderStructure.RAW)

        train_background_folder_path: str = CreateFolderStructure.create_folder(folder_name="train_background", folder_file_path=raw_folder_path)
        test_background_folder_path: str = CreateFolderStructure.create_folder(folder_name="test_background", folder_file_path=raw_folder_path)
        validation_background_folder_path: str = CreateFolderStructure.create_folder(folder_name="val_background", folder_file_path=raw_folder_path)

        train_folder_structure["train_background"] = train_background_folder_path
        train_folder_structure["test_background"] = test_background_folder_path
        train_folder_structure["validation_background"] = validation_background_folder_path

        return train_folder_structure, test_folder_structure, validation_folder_structure



    @classmethod
    def create_new_dataset_folder(cls, root_folder_file_path: str, dataset_folder_name: str) -> tuple[dict, dict, dict]:
        
        root_folder_file_path: str = CreateFolderStructure.create_folder(folder_name=dataset_folder_name, folder_file_path=root_folder_file_path)
        
        train_folder_structure: dict = CreateFolderStructure.create_folder_for_dataset_preparation(root_folder_file_path=root_folder_file_path, folder_name=CreateFolderStructure.IMAGES)
        test_folder_structure: dict = CreateFolderStructure.create_folder_for_dataset_preparation(root_folder_file_path=root_folder_file_path, folder_name=CreateFolderStructure.LABELS)
        validation_folder_structure: dict = CreateFolderStructure.create_folder_for_dataset_preparation(root_folder_file_path=root_folder_file_path, folder_name=CreateFolderStructure.RAW)
        

        return train_folder_structure, test_folder_structure, validation_folder_structure



    @classmethod
    def create_folder(cls, folder_name: str, folder_file_path: str) -> str:
        
        new_folder_path: str = os.path.join(folder_file_path, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        
        if os.path.exists(new_folder_path) == False:
            raise Exception("Unable to create folder due to unknown reason.")
        
        return new_folder_path


    @classmethod
    def create_folder_for_dataset_preparation(cls, root_folder_file_path: str, folder_name: str) -> dict[str, str]:
        new_root_folder_path: str = os.path.join(root_folder_file_path, folder_name)
        
        os.makedirs(new_root_folder_path, exist_ok=True)

        CreateFolderStructure.create_folder(CreateFolderStructure.FOLDER_TYPE[CreateFolderStructure.TRAIN], new_root_folder_path)
        CreateFolderStructure.create_folder(CreateFolderStructure.FOLDER_TYPE[CreateFolderStructure.TEST], new_root_folder_path)
        CreateFolderStructure.create_folder(CreateFolderStructure.FOLDER_TYPE[CreateFolderStructure.VALIDATION], new_root_folder_path)
        

        folder_structure_dictionary: dict[str, str] = {
            "images_root_folder": new_root_folder_path,
            "train_folder_path": os.path.join(new_root_folder_path, CreateFolderStructure.TRAIN),
            "test_folder_path": os.path.join(new_root_folder_path, CreateFolderStructure.TEST),
            "validation_folder_path": os.path.join(new_root_folder_path, CreateFolderStructure.VALIDATION)
        }

        return folder_structure_dictionary






