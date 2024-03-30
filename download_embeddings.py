import requests

from utils import create_directory

embeddings_list = [
    (
        "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin?download=true",
        "ip-adapter-faceid-plus_sd15.bin",
    ),
    (
        "https://civitai.com/api/download/models/77169?type=Model&format=PickleTensor",
        "BadDream.pt",
    ),
    (
        "https://civitai.com/api/download/models/42247?type=Model&format=Other",
        "realisticvision-negative-embedding.pt",
    ),
]

create_directory("embeddings")


def download_embeddings(url_list):
    for url, file_name in url_list:
        response = requests.get(url)
        if response.status_code == 200:
            # Extract filename from URL
            file_name = "embeddings/" + file_name

            with open(file_name, "wb") as f:
                f.write(response.content)
            print(f"File '{file_name}' downloaded successfully.")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")


download_embeddings(embeddings_list)
