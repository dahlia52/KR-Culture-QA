from huggingface_hub import hf_hub_download
import zipfile
import os

downloaded_zip = hf_hub_download(
    repo_id="jjae/ChromaDB-snowflake-arctic-embed-l-v2.0-ko-Kowiki-250611",
    filename="chroma_db.zip",
    repo_type="dataset"
)

target_root = "resource/retrieval_docs"

with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
