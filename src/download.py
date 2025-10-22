import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "../.cache_kagglehub"
path = kagglehub.dataset_download("nirmalsankalana/plant-diseases-training-dataset")

print("Dataset stored at:", path)