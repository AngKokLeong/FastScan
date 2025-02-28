from ultralytics import YOLO
import argparse
import os
import json
from pathlib import Path
import shutil
from ultralytics import settings
import file_manager

def train_yolo(pretrained_model_path: str, dataset_yaml_file_path: str, epochs: int = 10, learning_rate: float = 0.001, batch_size: int = 32) -> tuple[YOLO, dict]:
    fastscan_yolo_model: YOLO = YOLO(pretrained_model_path)


    # By setting project parameters in train(), the runs folder will be saved in that location
        # copy runs/train<#>/weights/best.pt to this folder path
    

        # This criteria is complete because the run folder will be generated in that defined location after the parameter pass into this function
            #  # /opt/ml/output/data for output the runs folder and other processed data to s3

    fastscan_yolov11_training_result: dict = fastscan_yolo_model.train(data=dataset_yaml_file_path, epochs=epochs, lr0=learning_rate, lrf=learning_rate, batch=batch_size)

    return fastscan_yolo_model, fastscan_yolov11_training_result

def validate_yolo_model(trained_yolo_model_file_path: str, dataset_yaml_file_path: str) -> dict:

    validation_fastscan_yolo_model: YOLO = YOLO(trained_yolo_model_file_path)
    results: dict = validation_fastscan_yolo_model.val(data=dataset_yaml_file_path, split="test")

    return results

    
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    # ... load from args.train and args.test, train a model, write model to args.model_dir.


    settings["runs_dir"] = "/opt/ml/output/runs"
    settings["datasets_dir"] = "/opt/ml/input/data/train/"



# SageMaker Training Toolkit 
# https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md  (information to help building the training script for determining input and output from the container in model training)

# SM_INPUT_DIR

    # SM_INPUT_DIR=/opt/ml/input/
    # /opt/ml/input
    # The path of the input directory, e.g. /opt/ml/input/. The input directory is the directory where SageMaker saves input data and configuration files before and during training.


# SM_INPUT_CONFIG_DIR
    # SM_INPUT_CONFIG_DIR=/opt/ml/input/config


        #The directory where standard SageMaker configuration files are located, e.g. /opt/ml/input/config/.

        #SageMaker training creates the following files in this folder when training starts:

        #    hyperparameters.json: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob request available in this file.
        #    inputdataconfig.json: You specify data channel information in the InputDataConfig parameter in a CreateTrainingJob request. Amazon SageMaker makes this information available in this file.
        #    resourceconfig.json: name of the current host and all host containers in the training.

        # For more information about these files, see: How Amazon SageMaker Provides Training Information (https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html)


# SM_MODEL_DIR
    # SM_MODEL_DIR=/opt/ml/model
    # When the training job finishes, the container and its file system will be deleted, with the exception of the /opt/ml/model and /opt/ml/output directories. 
    # Use /opt/ml/model to save the model checkpoints. 
    # These checkpoints will be uploaded to the default S3 bucket.



# Using this information from "How Amazon SageMaker AI provides Training Information" guide to find out the training data mountpoint
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html 



# Expect the input training fastscandataset folder can be placed in this mountpoint (/opt/ml/input/data/training)
    # Expected Path: /opt/ml/input/data/training/fastscandataset


# https://github.com/ultralytics/ultralytics/issues/3527 (Solution to move runs folder to another folder location)

    try:
        dataset_folder_location: str = file_manager.FileInformation.get_absolute_folder_location("/opt/ml/input/data/train/")
        yolov11n_file_path: str = file_manager.FileInformation.get_absolute_folder_location("/opt/ml/input/data/train/yolo11n.pt")
        fastscan_dataset_yaml_path: str = file_manager.FileInformation.get_absolute_folder_location("/opt/ml/input/data/train/fastscandataset.yaml")


        hyperparameters = json.loads(os.environ["SM_HPS"])
        epochs: int = int(hyperparameters["epochs"])
        batch_size: int = int(hyperparameters["batch-size"])
        learning_rate: float = float(hyperparameters["learning-rate"])

        trained_yolo_model, trained_yolo_model_result = train_yolo(pretrained_model_path=yolov11n_file_path, dataset_yaml_file_path=fastscan_dataset_yaml_path, 
                                                                    epochs=epochs,
                                                                    batch_size=batch_size,
                                                                    learning_rate=learning_rate)

        
        trained_yolo_model_best_pt_location: str = Path(str(trained_yolo_model_result.save_dir) + "/weights/best.pt").resolve()

        validate_yolo_model(trained_yolo_model_file_path=trained_yolo_model_best_pt_location, dataset_yaml_file_path=fastscan_dataset_yaml_path)



        # Output the data from the container
            # Using the guide "How Amazon SageMaker AI Processes Training Output" to find out the mount point for placing files in the folder for the container to output these files
            # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html


            # /opt/ml/model to place the trained model
        shutil.copyfile(src=trained_yolo_model_best_pt_location, dst="/opt/ml/model/best.pt", follow_symlinks=False)


    except Exception as ex:
        # /opt/ml/output/failure to write a file describing the failure reasons for this training
        with open("/opt/ml/output/failure/error.txt", "x") as textfile:
            textfile.write(ex)

