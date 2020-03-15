#!/Users/kaushik/opt/anaconda3/bin/python
"""Updates the model and writes the parameter files to GCP."""
import io
import gcp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd
import scraper
from typing import Text, List, Tuple
import win_probability


TEST_TEAMS = ['Afghanistan', 'Australia','Bangladesh',
         'England', 'India', 'Ireland',
         'New Zealand', 'Pakistan', 'South Africa',
         'Sri Lanka', 'West Indies', 'Zimbabwe']


def train_test_split(game_ids: List[int], ratio=0.8) -> Tuple[List[int], List[int]]:
    """Randomly splits the data into a test set and a train set

    :param game_ids:
    :param ratio:
    :return:
    """
    test_ids = []
    train_ids = []
    for game_id in game_ids:
        if np.random.random() <= 0.8:
            train_ids.append(game_id)
        else:
            test_ids.append(game_id)
    return test_ids, train_ids


def filter_test_teams(ball_by_ball: pd.DataFrame, match_summary: pd.DataFrame, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    match_summary = match_summary[match_summary['1st_Innings'].isin(TEST_TEAMS)]
    match_summary = match_summary[match_summary['2nd_Innings'].isin(TEST_TEAMS)]

    ball_by_ball = ball_by_ball[ball_by_ball.Game_Id.isin(match_summary.Game_Id.unique())]
    return ball_by_ball, match_summary


def get_data_frame(match_type: Text) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bbb = gcp.download_data_frame('ball-by-ball', match_type)
    ms = gcp.download_data_frame('match-summary', match_type)
    return bbb, ms


def update_model(game_type: Text) -> None:
    """Trains the model and uploads the data to GCP."""
    assert game_type in ['ODI', 'T20']
    if game_type == 'T20':
        print('Updating T20I games...')
        scraper.update_t20i()
        print('Updating IPL games...')
        scraper.update_ipl()
        # want to combine International and IPL
        # TODO: Later get all the leagues data
        t20_bbb, t20_ms = get_data_frame('T20I')
        ipl_bbb, ipl_ms = get_data_frame('IPL')
        t20_bbb, t20_ms = filter_test_teams(t20_bbb, t20_ms)
        ball_by_ball = t20_bbb.append(ipl_bbb)
        match_summary = t20_ms.append(ipl_ms)
        match_summary = match_summary[~match_summary.Reduced_Over]
        ball_by_ball = ball_by_ball[ball_by_ball.Game_Id.isin(match_summary.Game_Id.unique())]
    else:
        print('Updating ODI games...')
        scraper.update_odi()
        bbb, ms = get_data_frame('ODI')
        ball_by_ball, match_summary = filter_test_teams(bbb, ms)
    game_ids = list(match_summary.Game_Id.unique())
    test_ids, train_ids = train_test_split(game_ids)

    train_bbb = ball_by_ball[ball_by_ball.Game_Id.isin(train_ids)]
    train_ms = match_summary[match_summary.Game_Id.isin(train_ids)]

    max_over = 50 if game_type == 'ODI' else 20
    print('Training...')
    parameters = win_probability.fit(train_bbb, train_ms, max_over)

    # Draw the graphs for test data
    print('Plots for the testing data...')
    params_1 = parameters[parameters.innings == 1]
    params_2 = parameters[parameters.innings == 2]
    pdf = PdfPages('wp.pdf')
    for test_id in test_ids:
        try:
            scorecard = ball_by_ball[ball_by_ball.Game_Id == test_id]
            summary = match_summary[match_summary.Game_Id == test_id]
            fi, si = win_probability.predict_game(scorecard, max_over, params_1, params_2)
            fig = win_probability.plot_game(fi, si, summary, max_over)
            pdf.savefig(fig)
            plt.close()
        except:
            continue
    pdf.close()

    # Create the WP Matrix
    print('Creating the WP Matrix...')
    matrix_1 = open('first_innings_matrix.txt', 'w')
    matrix_2 = open('second_innings_matrix.txt', 'w')
    max_runs = 500 if game_type == 'ODI' else 300
    for prediction in win_probability.win_probability_matrix_generator(params_1,
                                                                       params_2,
                                                                       max_over,
                                                                       max_runs):
        text_1, text_2 = win_probability.win_probability_matrix_generator_cricmetric_format(prediction, game_type == 'ODI')
        matrix_1.write(text_1)
        matrix_1.write("\n")

        matrix_2.write(text_2)
        matrix_2.write("\n")
    matrix_1.close()
    matrix_2.close()

    # Write to the GCP Buckets
    # parameters, wp.pdf, matrix_1.txt, matrix_2.txt
    print('Uploading...')
    bucket_name = 'odi-model' if game_type == 'ODI' else 't20-model'
    blob = gcp.get_blob(bucket_name, 'parameters.csv')
    data = io.StringIO(parameters.to_csv(index=False, encoding='ISO-8859-1'))
    print('Uploading to blob:{0} in bucket: {1}'.format('parameters.csv', bucket_name))
    blob.upload_from_file(data)

    blob = gcp.get_blob(bucket_name, 'wp.pdf')
    print('Uploading to blob:{0} in bucket: {1}'.format('wp.pdf', bucket_name))
    blob.upload_from_filename("wp.pdf")

    blob = gcp.get_blob(bucket_name, 'first_innings_matrix.txt')
    print('Uploading to blob:{0} in bucket: {1}'.format('first_innings_matrix.txt', bucket_name))
    blob.upload_from_filename("first_innings_matrix.txt")

    blob = gcp.get_blob(bucket_name, 'second_innings_matrix.txt')
    print('Uploading to blob:{0} in bucket: {1}'.format('second_innings_matrix.txt', bucket_name))
    blob.upload_from_filename("second_innings_matrix.txt")

    # Delete the temporary files
    print('Cleaning up...')
    os.remove("wp.pdf")
    os.remove("first_innings_matrix.txt")
    os.remove("second_innings_matrix.txt")


if __name__ == '__main__':
    print('Updating the T20 model...')
    update_model('T20')

    print('Updating the ODI model...')
    update_model('ODI')
