import os
import kagglehub

if not os.path.exists("../.cache_kagglehub"):
    os.makedirs("../.cache_kagglehub")

os.environ["KAGGLEHUB_CACHE"] = "../.cache_kagglehub"
path = kagglehub.dataset_download("nirmalsankalana/plant-diseases-training-dataset")

print("Dataset sementara tersimpan di:", path)