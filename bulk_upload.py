"""Uploads all the data to google cloud storage bucket. Just one time use only."""

from google.cloud import storage
import functools
import glob
import pandas as pd
import re
import toolz
from toolz.curried import map, filter
from typing import Text, Optional


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def upload(location):
    def match(string: Text, pattern: Text) -> Optional[Text]:
        out = re.match(pattern, string)
        if out:
            return out.groups()[0]
        return None

    def upload_single_file(source_file_name, bucket_name, pattern):
        game_id = match(source_file_name, pattern)
        if game_id:
            destination_blob_name = '{0}/{1}.csv'.format(location, game_id)
            upload_blob(bucket_name, source_file_name, destination_blob_name)

    ball_by_ball_file_names = glob.glob('{0}/data/ball_by_ball/*.csv'.format(location))
    match_summary_file_names = glob.glob('{0}/data/match_summary/*.csv'.format(location))

    ball_by_ball_match_string = "{0}/data/ball_by_ball/([0-9]*).csv".format(location)
    match_summary_match_string = "{0}/data/match_summary/([0-9]*).csv".format(location)

    list(map(functools.partial(upload_single_file, bucket_name='ball-by-ball', pattern=ball_by_ball_match_string),
             ball_by_ball_file_names))
    list(map(functools.partial(upload_single_file, bucket_name='match-summary', pattern=match_summary_match_string),
             match_summary_file_names))


def consolidate_and_upload(location):

    def consolidate(file_names):
        return toolz.pipe(file_names, map(pd.read_csv), filter(lambda x: len(x) > 0), list, pd.concat)

    ball_by_ball_file_names = glob.glob('{0}/data/ball_by_ball/*.csv'.format(location))
    match_summary_file_names = glob.glob('{0}/data/match_summary/*.csv'.format(location))

    ball_by_ball_df = consolidate(ball_by_ball_file_names)
    match_summary_df = consolidate(match_summary_file_names)

    bbb_loc = '{0}/data/all_ball_by_ball.csv'.format(location)
    ball_by_ball_df.to_csv(bbb_loc, index=False)
    ms_loc = '{0}/data/all_match_summary.csv'.format(location)
    match_summary_df.to_csv(ms_loc, index=False)
    upload_blob('ball-by-ball', bbb_loc, '{0}/all.csv'.format(location))
    upload_blob('match-summary', ms_loc, '{0}/all.csv'.format(location))
