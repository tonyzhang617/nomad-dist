from datasets import load_dataset
import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


# Download wikitext-2
download_and_unzip('https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip')

# Download PTB
os.makedirs("./ptb", exist_ok=True)

for split in ['train', 'validation', 'test']:
    dataset = load_dataset("ptb_text_only", split=split)

    sentences = [
        dataset[i]['sentence'] for i in range(dataset.num_rows)
    ]

    with open(f'./ptb/{split}.txt', 'w') as file:
        file.write('\n'.join(sentences))
