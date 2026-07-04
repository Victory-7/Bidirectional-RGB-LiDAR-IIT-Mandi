import requests
import zipfile
import io
import os
import shutil

url = "https://github.com/CompVis/taming-transformers/archive/refs/heads/master.zip"
output_dir = "taming-transformers"

try:
    print("Downloading taming-transformers...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(".")

    extracted = "taming-transformers-master"

    if os.path.exists(extracted):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.rename(extracted, output_dir)

    print(f"Success! Repository downloaded to {output_dir}")

except Exception as e:
    print(f"Error: {e}")
