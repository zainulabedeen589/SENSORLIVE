
# import os
# class S3Sync:


#     def sync_folder_to_s3(self,folder,aws_buket_url):
#         command = f"aws s3 sync {folder} {aws_buket_url} "
#         os.system(command)

#     def sync_folder_from_s3(self,folder,aws_bucket_url):
#         command = f"aws s3 sync  {aws_buket_url} {folder} "
#         os.system(command)






# import os

# class S3Sync:

#     def sync_folder_to_s3(self, folder, aws_bucket_url):
#         # Fixed typo in variable name: aws_buket_url -> aws_bucket_url
#         # Removed trailing space inside the string before the closing quote
#         command = f"aws s3 sync {folder} {aws_bucket_url}"
#         os.system(command)

#     def sync_folder_from_s3(self, folder, aws_bucket_url):
#         # Removed trailing space inside the string before the closing quote
#         command = f"aws s3 sync {aws_bucket_url} {folder}"
#         os.system(command)







import subprocess

class S3Sync:

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        command = ["aws", "s3", "sync", folder, aws_bucket_url]
        subprocess.run(command, check=True)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        command = ["aws", "s3", "sync", aws_bucket_url, folder]
        subprocess.run(command, check=True)
