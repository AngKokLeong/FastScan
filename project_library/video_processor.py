import cv2
import os
import numpy

import project_library.image_processor
import project_library.IMAGE_CONFIGURATION

import copy
from pathlib import Path

import project_library.video_processor




class VideoExtraction:

    @classmethod
    def process_video_files_within_folder_to_image_data(cls, video_folder_path: str, resize_dimension: int) -> list[list[numpy.ndarray]]:

        image_data_collection: list[list[numpy.ndarray]] = []

        #for each video file
        for video_file in os.listdir(video_folder_path):
            # invoke extract_image_data_from_video()    
            # append the list[numpy.ndarray] to another list 
                # list[list[numpy.ndarray]]
            print(f"Processing {video_file} ....")

            image_data_collection.append(VideoExtraction.extract_image_data_with_square_dimension_from_video(os.path.join(video_folder_path, video_file), resize_dimension))
            
            print(f"Complete Processing {video_file}.")

        return image_data_collection


    @classmethod
    def extract_image_data_with_square_dimension_from_video(cls, source_video_file_path: str, resize_dimension: int) -> list[numpy.ndarray]:

        # determine if the video file exists
        # This constructor opens the video file
        video_capture_object: cv2.VideoCapture = cv2.VideoCapture(filename=source_video_file_path, apiPreference=cv2.CAP_ANY)

        if video_capture_object.isOpened() == False:
            raise Exception("Unable to open the video file")

        # determine if the video has any file size

        video_width: int = int(video_capture_object.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height: int = int(video_capture_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_per_second: float = video_capture_object.get(cv2.CAP_PROP_FPS)
        number_of_frame: int = video_capture_object.get(cv2.CAP_PROP_FRAME_COUNT)

        image_data_list: list[numpy.ndarray] = []

        while True:

            # Grab the video frame 
            retrieving_next_frame_status: bool = video_capture_object.grab()
            
            # If the video frame cannot be obtained then close the video object and exit the loop
            if retrieving_next_frame_status == False:
                video_capture_object.release()
                break
            
            retrieved_image: cv2.Mat = None
            image_retrieval_status: bool = False
            
            # Retrieve the current frame
            image_retrieval_status, retrieved_image = video_capture_object.retrieve()

            if image_retrieval_status == False:
                raise Exception("There is an error during the image retrieval from the video.")

            
            # Resize the image to square dimension while ensuring image aspect ratio
            resized_image: numpy.ndarray = project_library.image_processor.ResizeImage.resize_image_to_square_dimension(retrieved_image, resize_dimension)

            processed_image: numpy.ndarray = project_library.image_processor.ImageColorSpaceConversion.convert_image_into_RGB_space(image_data=resized_image)


            # Add the image data to a list
            image_data_list.append(processed_image)
        
        return image_data_list
    
    @classmethod
    def extract_Nth_image_from_image_collection(cls, image_data_collection: list[list[numpy.ndarray]], N_value:int) -> list[list[numpy.ndarray]]:

        filtered_image_data_collection: list[list[numpy.ndarray]] = []

        for image_data_list in image_data_collection:
            filter_image_data_list: list[numpy.ndarray] = project_library.video_processor.VideoExtraction.extract_Nth_image_data(image_data_list, N_value)

            filtered_image_data_collection.append(filter_image_data_list)

        return filtered_image_data_collection


    @classmethod
    def extract_Nth_image_data(cls, image_data_list: list[numpy.ndarray], N_value: int) -> list[numpy.ndarray]:

        #Create another copy of the list
        image_data_list_copy = copy.deepcopy(image_data_list)
        filtered_image_data_list: list[numpy.ndarray] = []

        for index, image_data in enumerate(image_data_list_copy):

            if index % N_value == 0:
                filtered_image_data_list.append(image_data)
        
        # Calculate the optimal number of images to capture
        
        return filtered_image_data_list
        


    @classmethod
    def remove_Nth_image_data(cls, image_data_list: list[numpy.ndarray], N_value: int, FPS: int) -> list[numpy.ndarray]:
        
        # determine the total number of image data in a list
        total_number_of_images: int = len(image_data_list)
        
        # determine the FPS in the video


        # determine the length of the video

        #Create another copy of the list
        image_data_list_copy = copy.deepcopy(image_data_list)


        for index, data in enumerate(image_data_list_copy):

            if index % N_value == 0:
                image_data_list_copy.pop(index)
        
        # Calculate the optimal number of images to capture
        
        return image_data_list_copy
    # need to find a way to selectively retrieve the image data based on every 5 images or every N images

    @classmethod
    def output_image_data_collection_to_folder(cls, image_data_collection: list[list[numpy.ndarray]], destination_folder_path: str) -> None:
        
        # Purpose: Output the image data to a single folder

        # Determine if the destination_folder_path exists
        relative_destination_folder_path: Path = Path(destination_folder_path)
        absolute_destination_folder_path: Path = relative_destination_folder_path.resolve()

        # create the custom raw image folder at the project root folder
        os.makedirs(absolute_destination_folder_path, exist_ok=True)
        
        project_library.image_processor.OutputImages.create_images_from_collection_to_folder(image_data_collection=image_data_collection, destination_folder_path=absolute_destination_folder_path, image_label="raw")
            
        return None

    @classmethod
    def output_image_data_to_folder(cls, image_data_list: list[numpy.ndarray], destination_folder_path: str, image_type: str = "") -> None:
        # Invoke the method to output the image data from numpy array to folder
        project_library.image_processor.OutputImages.create_images_to_a_folder(image_data_list, destination_folder_path, image_type)

        return None