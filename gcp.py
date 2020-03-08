"""Functions to interact with GCP."""
from google.cloud import storage
import io
import pandas as pd
from typing import Text


CREDENTIALS = '/Users/kaushik/.gcp_credentials/cricket_analytics.json'


def get_client():
    return storage.Client.from_service_account_json(CREDENTIALS)


def get_blob(bucket_name: Text, blob_name: Text) -> storage.Blob:
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob


def upload_data_frame(bucket_name: Text, blob_name: Text, df: pd.DataFrame) -> None:
    """Uploads the data frame to the bucket."""
    blob = get_blob(bucket_name, blob_name)
    data = io.StringIO(df.to_csv(index=False))
    print('Uploading to blob:{0} in bucket: {1}'.format(blob_name, bucket_name))
    blob.upload_from_file(data)


def download_data_frame(bucket_name: Text, blob_name: Text) -> pd.DataFrame:
    blob = get_blob(bucket_name, blob_name)
    print('Downloading Blob:{0} in bucket: {1}'.format(blob_name, bucket_name))
    data = io.BytesIO(blob.download_as_string())
    return pd.read_csv(data)
