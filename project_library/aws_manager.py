import boto3

class S3:

   
    s3_client = boto3.client("s3")

    @classmethod
    def list_s3_keys(cls, bucket_name: str, prefix: str = '') -> list:
        """
            List all keys in an S3 bucket with given prefix
    
            Args:
                bucket_name (str): Name of the S3 Bucket
                prefix (str): Prefix to filter objects (optional)
        
            Returns:
                list: List of S3 Keys
        """
        try:
            s3_client = cls.s3_client    
            keys: list = []
    
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        keys.append(obj["Key"])
            
            if not keys:
                print(f"No objects found in bucket '{bucket_name}'")
    
            return keys
        except s3_client.exceptions.NoSuchBucket:
            print(f"Bucket '{bucket_name}' does not exist")
            return []
        except Exception as e:
            print(f"Error listing objects: {str(e)}")
            return []


    @classmethod
    def download_all_files(cls, bucket_name: str, prefix: str, destination_folder_path: str) -> None:

        s3_file_list: list = S3.list_s3_keys(bucket_name=bucket_name, prefix=prefix)

        for s3_file in s3_file_list:
            print(f"Processing {s3_file}")
            cls.s3_client.download_file(bucket_name, s3_file, destination_folder_path)

