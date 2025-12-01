import os
import kagglehub

if not os.path.exists("../.cache_kagglehub"):
    os.makedirs("../.cache_kagglehub")

os.environ["KAGGLEHUB_CACHE"] = "../.cache_kagglehub"
path = kagglehub.dataset_download("aryashah2k/mango-leaf-disease-dataset")

print("Cache saved at:", path)