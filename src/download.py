import kagglehub

kagglehub.set_cache_dir("./cache_kagglehub")
path = kagglehub.dataset_download("nirmalsankalana/plant-diseases-training-dataset")

print("Path to dataset files:", path)