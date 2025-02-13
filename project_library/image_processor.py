import cv2
import numpy
import os
import matplotlib.pyplot as plt

class ImageProcessPipeline:

    @staticmethod
    def process_images_in_folder(image_folder_to_process: str) -> list:
        
        processed_image_data_list: list = []

        for image_file_path in os.listdir(image_folder_to_process):

            image_data = ImageColorSpaceConversion.convert_image_to_grayscale_format(image_file_path)
            image_data = ResizeImage.resize_image(image=image_data, image_size=(100, 100))

            processed_image_data_list.append(image_data)
        
        return processed_image_data_list


class OutputImages:

    '''
        Output Image for a level of folder
    '''

    @classmethod
    def create_images_to_a_folder(cls, image_data_list: list, destination_folder_path: str, image_label: str = "") -> None:

        for index, image_data in enumerate(image_data_list):
            cv2.imwrite(filename=os.path.join(destination_folder_path, str(image_label) + str(index)) + str(".jpg"), img=image_data)

    @classmethod
    def create_images_from_collection_to_folder(cls, image_data_collection: list[list[numpy.ndarray]], destination_folder_path: str, image_label: str = "") -> None:
        image_index: int = 0
        for image_data_list in image_data_collection:
            for image_data in image_data_list:

                # This conversion restore the natural colors back
                # By Default each time 
                processed_image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

                cv2.imwrite(filename=os.path.join(destination_folder_path, str(image_label) + "_" + str(image_index)) + str(".jpg"), img=processed_image_data)

                image_index += 1
                
        return None
        


class DisplayImage:

    def display_image_from_numpy_array(image_data) -> None:
        plt.imshow(image_data)
        plt.axis('off')
        plt.show()


class ImageDataRetrieval:

    @classmethod
    def retrieve_image_data_from_file_path(cls, image_file_path: str) -> numpy.ndarray:
        image_data: numpy.ndarray = cv2.imread(image_file_path, flags=cv2.IMREAD_COLOR_RGB)
        
        return image_data

    @classmethod
    def retrieve_image_data_from_folder(cls, image_folder_path: str) -> list[numpy.ndarray]:
        
        image_data_list: list[numpy.ndarray] = []

        for image_file_path in os.listdir(image_folder_path):
            
            full_image_file_path: str = os.path.join(image_folder_path, image_file_path)
            
            image_data_list.append(ImageDataRetrieval.retrieve_image_data_from_file_path(full_image_file_path))


        return image_data_list


class ImageColorSpaceConversion:


    @classmethod
    def convert_image_into_RGB_space(cls, image_data: numpy.ndarray) -> numpy.ndarray:
        if image_data.size == 0:
            raise Exception("There is no image data given")
        
        processed_image: numpy.ndarray = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        return processed_image
    
    @staticmethod
    def retrieve_and_convert_image_to_grayscale_format(image_file_path: str) -> numpy.ndarray:
        
        # Load the image in grayscale
        image = cv2.imread(image_file_path, flags=cv2.IMREAD_COLOR_RGB)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #Image format problem resulting in images become purple or other colours # https://stackoverflow.com/questions/59600937/cv2-cvtcolorimg-cv2-color-bgr2rgb-not-working
        
        if type(image) != numpy.ndarray and image == None:
          return Exception("Empty file")
        
        return image
    
    @staticmethod
    def retrieve_and_convert_images_into_grayscale_format(image_folder_path: str) -> list:
        
        image_data_list: list = []

        for image_file_path in os.listdir(image_folder_path):
            
            full_image_file_path: str = os.path.join(image_folder_path, image_file_path)
            
            image_data_list.append(ImageColorSpaceConversion.retrieve_and_convert_image_to_grayscale_format(full_image_file_path))

        return image_data_list
        

    @staticmethod
    def convert_image_into_grayscale_format(image_data: numpy.ndarray) -> numpy.ndarray:

        if type(image_data) != numpy.ndarray and image_data == None:
            raise Exception("Empty file")
        
        converted_image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #Image format problem resulting in images become purple or other colours # https://stackoverflow.com/questions/59600937/cv2-cvtcolorimg-cv2-color-bgr2rgb-not-working
        
        return converted_image_data
    
    @staticmethod
    def convert_images_into_grayscale_format(image_data_list: list[numpy.ndarray]) -> list[numpy.ndarray]:
        
        if len(image_data_list) == 0:
            raise Exception("There is no image data in the list.")

        converted_image_data_list: list[numpy.ndarray] = []

        for image_data in image_data_list:
            converted_image_data = ImageColorSpaceConversion.convert_image_into_grayscale_format(image_data)

            converted_image_data_list.append(converted_image_data)

        return converted_image_data_list



class ResizeImage:
    
    @classmethod
    def resize_image_by_file_path(cls, image_file_path: str, image_size: tuple[int, int]) -> numpy.ndarray:
        retrieved_image = cv2.imread(image_file_path)

        resized_image = cv2.resize(src=retrieved_image, dsize=image_size)
        
        return resized_image
    
    @classmethod
    def resize_image(cls, image: numpy.ndarray, image_size: tuple[int, int]) -> numpy.ndarray:

        resized_image = cv2.resize(src=image, dsize=image_size)
        return resized_image
    
    @classmethod
    def resize_image_to_square_dimension(cls, image: numpy.ndarray, image_size_to_resize: int) -> numpy.ndarray:

        height, width = image.shape[: 2]

        aspect_ratio: float = width / height

        new_image_width: int = 0
        new_image_height: int = 0

        if aspect_ratio > 1:
            new_image_width = image_size_to_resize
            new_image_height = int(image_size_to_resize / aspect_ratio)
        elif aspect_ratio <= 1:
            new_image_height = image_size_to_resize
            new_image_width = int(image_size_to_resize * aspect_ratio)


        resized_image = cv2.resize(src=image, dsize=(new_image_width, new_image_height))


        canvas = numpy.zeros((image_size_to_resize, image_size_to_resize, 3), dtype=numpy.uint8)

        x_coordinate_offset = (image_size_to_resize - new_image_width) // 2
        y_coordinate_offset = (image_size_to_resize - new_image_height) // 2

        canvas[y_coordinate_offset: y_coordinate_offset + new_image_height, x_coordinate_offset: x_coordinate_offset + new_image_width] = resized_image

        return canvas

    @classmethod
    def resize_images(cls, image_data_list: list[numpy.ndarray], image_size: tuple[int, int]) -> numpy.ndarray:

        resized_image_data_list: list[numpy.ndarray] = []

        for image_data in image_data_list:
            resized_image_data_list.append(ResizeImage.resize_image(image=image_data, image_size=image_size))

        return resized_image_data_list
    
