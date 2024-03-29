"""Fits the Logistic Regression Model to the data."""
__author__ = 'kaushik'

from collections import namedtuple
from itertools import chain, product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from toolz import groupby, pipe
from toolz.curried import map, filter
from typing import List, Tuple, Iterator, Dict, Text
import warnings

pd.set_option('mode.chained_assignment', None)
MAX_WICKETS = 10
EPSILON_RUNS = 0.01
EPSILON_WICKETS = 0.01

Feature = namedtuple("Feature", ["Innings", "Over", "Ball", "Runs", "Target", "RunRate", "Wickets"])
Target = namedtuple("Target", ["Target"])
Feature_Target_Pair = Tuple[Feature, Target]
Parameter = namedtuple("Parameter", ["over", "intercept", 'run_rate', 'wicket', 'mu', 'sigma'])
Prediction = namedtuple("Prediction", ["Innings", "Over", "Ball", "Runs", "Target",
                                       "RunRate", "Wickets", "win_probability", "score", "score_lo", "score_hi"])

# Related to reading the data


def validate_inputs(ball_by_ball_data: pd.DataFrame, innings: int, max_over: int, result: int) -> None:
    """Validates the input data frame

    :param ball_by_ball_data:  data frame with ball by ball data
    :param innings:
    :param max_over:
    :param result:
    :return:
    """
    assert innings in [1, 2]
    assert max_over in [50, 20]
    assert result in [0, 1]
    assert 'Runs' in ball_by_ball_data.columns
    assert 'Over' in ball_by_ball_data.columns
    assert 'Wicket' in ball_by_ball_data.columns
    assert 'Ball' in ball_by_ball_data.columns
    assert 'Target' in ball_by_ball_data.columns
    assert 'Run_Rate' in ball_by_ball_data.columns
    assert 'Required_Run_Rate' in ball_by_ball_data.columns
    assert 'Innings' in ball_by_ball_data.columns


def process_innings_from_scorecard(ball_by_ball_data: pd.DataFrame, innings: int) -> pd.DataFrame:
    """Extracts data fo the relevant innings from the ball by ball data and adds over number, balls, total runs etc.

    :param ball_by_ball_data: A data frame with the ball by ball data
    :param innings: The innings for which we want to extract data
    :return: The processed innings data frame
    """
    assert innings in [1, 2]
    inn = ball_by_ball_data[ball_by_ball_data.Innings == innings].copy(deep=True)
    inn['BallN'] = (inn.Ball.apply(lambda x: x - x // 1)) * 100
    inn['BallN'] = inn.BallN.apply(np.round)
    inn['OverKey'] = inn.Over - 1 + inn.BallN / 6
    inn.sort_values(by=['Over', 'BallN'], inplace=True)
    inn['cumRuns'] = inn.Runs.cumsum()
    inn['cumWickets'] = inn.Wicket.cumsum()
    return inn


def get_features(state, innings: int, max_over: int) -> Feature:
    """ Extracts the feature from the ball by ball data.

    :param state: A namedtuple containing one row of data from the ball by ball data
    :param innings: The innings for which we want to extract data
    :param max_over: The maximum overs in the game (50 for ODI, 20 for T20)
    :return: Feature's from the game state to be used in the modeling
    """
    over = state.Over
    ball = state.BallN
    run_rate = state.Run_Rate if innings == 1 else state.Required_Run_Rate
    wickets = state.cumWickets
    runs = state.cumRuns
    target = state.Target

    if innings == 2 and over == max_over and ball == 6:
        run_rate = 36.0 if state.cumRuns < state.Target else run_rate

    feature = Feature(innings, over, ball, runs, target, run_rate, wickets)
    return feature


def create_feature_vector(ball_by_ball_data: pd.DataFrame, innings: int, max_over: int) -> List[Feature]:
    """Creates the feature vector for every ball in the game

    :param ball_by_ball_data: A dataframe with the ball by ball data
    :param innings:  The innings for which we want to extract the feature
    :param max_over: The maximum overs in the game (50 for ODI, 20 for T20)
    :return: A list of Feature (one for every ball in the innings)
    """
    validate_inputs(ball_by_ball_data, innings, max_over, 1)
    inn = process_innings_from_scorecard(ball_by_ball_data, innings)
    if len(inn) == 0:
        warnings.warn('Empty data frame')
        return []
    return [get_features(state, innings, max_over) for state in inn.itertuples()]


def create_feature_and_target_vector(ball_by_ball_data: pd.DataFrame,
                                     result: int, innings: int, max_over: int) -> List[Feature_Target_Pair]:
    """Creates the feature and target vector for every over in the game.

    :param ball_by_ball_data: A dataframe with the ball by ball data
    :param result: 0 if the first innings team won, 1 if the second innings team won
    :param innings: The innings for which we want to extract the feature
    :param max_over: The maximum overs in the game (50 for ODI, 20 for T20)
    :return: A list of Tuples. Each tuple consists of a feature and the target
    """

    def get_features_and_target_from_state(state):
        feature = get_features(state, innings, max_over)
        target = Target(first_innings_target) if innings == 1 else Target(second_innings_target)
        return feature, target

    validate_inputs(ball_by_ball_data, innings, max_over, result)
    inn = process_innings_from_scorecard(ball_by_ball_data, innings)
    second_innings_target = result
    first_innings_target = inn.tail(1).Run_Rate.values[0]
    inn = inn[inn.BallN == inn.BallN.min()]
    if len(inn) == 0:
        warnings.warn('Empty data frame')
        return []
    return [get_features_and_target_from_state(state) for state in inn.itertuples()]

# Fit the model


def fit_model(features_and_targets: Iterator[Feature_Target_Pair], innings: int) -> pd.DataFrame:
    """Fits the model that predicts win probability.

    :param features_and_targets:  A List of the feature target pairs that consist of the data that we wish to model.
    :param innings: The innings that we wish to model.
    :return: A data frame with the parameters of the model.
    """
    assert innings in [1, 2]

    def fit_model_for_over(pair: Tuple[int, List[Feature_Target_Pair]]) -> Parameter:
        """Helper function that fits the model for data from a single over.
        Note that this method uses variables from the outer scope.

        :param pair: A tuple - first element is the over and the second element is the data
        :return: Parameter fit for that over
        """

        def feature_vector(feature_target_pair: Feature_Target_Pair) -> np.array:
            """Converts the feature into a numpy array.

            :param feature_target_pair:
            :return: numpy array
            """
            feature, _ = feature_target_pair
            log_run_rate = np.log(feature.RunRate + EPSILON_RUNS)
            log_wicket = np.log((MAX_WICKETS - feature.Wickets) + EPSILON_WICKETS)
            return np.array([log_run_rate, log_wicket])

        def target_vector(feature_target_pair: Feature_Target_Pair) -> np.array:
            """Converts the target into a numpy array.

            :param feature_target_pair:
            :return: numpy array
            """
            _, target = feature_target_pair
            return np.array(np.log(target.Target)) if innings == 1 else np.array(target.Target)

        def fit_first_innings(x_matrix: np.array, y: np.array, over: int) -> Parameter:
            """Fits the linear regression model to predict the end of the innings run rate.

            :param x_matrix: Matrix of regressors
            :param y: Vector of observed values
            :param over: Over for which we are predicting
            :return: Parameters of that over
            """
            lr = LinearRegression()
            lr.fit(x_matrix, y)
            predictions = lr.predict(x_matrix)
            errors = predictions - y
            return Parameter(over, lr.intercept_, lr.coef_[0], lr.coef_[1], np.mean(errors), np.std(errors))

        def fit_second_innings(x_matrix: np.array, y: np.array, over: int) -> Parameter:
            """Fits the logistic regression model to predict the winner at the end of second innings.

            :param x_matrix: Matrix of regressors
            :param y: ector of observed values
            :param over: Over for which we are predicting
            :return: Parameters of that over
            """
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(x_matrix, y)
            return Parameter(over, lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1], 0, 0)

        over_num, feature_target_pair_list = pair
        features = np.array([feature_vector(pair) for pair in feature_target_pair_list])
        targets = np.array([target_vector(pair) for pair in feature_target_pair_list])
        return fit_first_innings(features, targets, over_num) if innings == 1 else \
            fit_second_innings(features, targets, over_num)

    def group_by_overs() -> Iterator[Tuple[int, List[Feature_Target_Pair]]]:
        return groupby(lambda x: x[0].Over, features_and_targets).items()

    parameters = [fit_model_for_over(pair) for pair in group_by_overs()]
    return pd.DataFrame(parameters, columns=['overs', 'intercept',
                                             'log_rr_param', 'log_wickets_param', 'mu', 'error_std'])


def predict_state(state: Feature, max_over: int, params_1: Dict[int, Parameter],
                  params_2: Dict[int, Parameter]) -> Prediction:
    """Predicts the win probability from the current state of the game

    :param state: The state of the game
    :param max_over:  The max overs in the game
    :param params_1: 1st innings parameters
    :param params_2: 2nd innings parameters
    :return: The prediction for the given state
    """

    over = state.Over
    innings = state.Innings
    run_rate, wickets = state.RunRate, state.Wickets
    log_wickets = np.log((MAX_WICKETS - wickets) + EPSILON_WICKETS)
    log_rr = np.log(run_rate + EPSILON_RUNS)

    if innings == 1:
        params = params_1[over]
        intercept = params.intercept
        log_rr_param = params.run_rate
        log_wickets_param = params.wicket
        std = params.sigma
        prediction = intercept + log_rr_param * log_rr + log_wickets_param * log_wickets
        # make the second innings prediction
        params_predict = params_2[1]
        intercept = params_predict.intercept
        log_rr_param = params_predict.run_rate
        log_wickets_param = params_predict.wicket
        prediction = np.log(np.exp(prediction) + EPSILON_RUNS)
        reply = intercept + log_rr_param * prediction + log_wickets_param * np.log(10 + EPSILON_WICKETS)
        win_probability = np.exp(reply) / (1 + np.exp(reply))
        predicted_score = int(np.exp(prediction) * max_over)
        predicted_score_lo = int(np.exp(prediction - std) * max_over)
        predicted_score_hi = int(np.exp(prediction + std) * max_over)
        return Prediction(1, state.Over, state.Ball, state.Runs, state.Target,
                          state.RunRate, state.Wickets,
                          1 - win_probability, predicted_score, predicted_score_lo, predicted_score_hi)
    else:
        params = params_2[over]
        intercept = params.intercept
        log_rr_param = params.run_rate
        log_wickets_param = params.wicket
        prediction = intercept + log_rr_param * log_rr + log_wickets_param * log_wickets
        win_probability = np.exp(prediction) / (1 + np.exp(prediction))
        predicted_score = -1
        predicted_score_lo = -1
        predicted_score_hi = -1
        return Prediction(2, state.Over, state.Ball, state.Runs, state.Target,
                          state.RunRate, state.Wickets,
                          1 - win_probability, predicted_score, predicted_score_lo, predicted_score_hi)


def fit(ball_by_ball_data: pd.DataFrame, match_summary_data: pd.DataFrame, max_over: int) -> pd.DataFrame:
    """Gets the parameter fit for the games in the data."""
    def get_game_data(game_id) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ball_by_ball_df = ball_by_ball_data[ball_by_ball_data.Game_Id == game_id]
        match_summary_df = match_summary_data[match_summary_data.Game_Id == game_id]
        return ball_by_ball_df, match_summary_df

    def extract_features(game_data: Tuple[pd.DataFrame, pd.DataFrame]) -> List[List[Feature_Target_Pair]]:
        bd, md = game_data
        if md.Reduced_Over.values[0]:
            return []
        result = md.Winning_Innings.values[0] - 1

        if result not in [0, 1]:
            return []

        f1 = create_feature_and_target_vector(bd, result, 1, max_over)
        f2 = create_feature_and_target_vector(bd, result, 2, max_over)

        if f1 and f2:
            return [f1, f2]
        return []

    match_summary_data = match_summary_data[~match_summary_data.Reduced_Over]
    bbb_game_ids = set(ball_by_ball_data.Game_Id.unique())
    ms_game_ids = set(match_summary_data.Game_Id.unique())
    game_ids = bbb_game_ids.intersection(ms_game_ids)
    data = pipe(game_ids, map(get_game_data), map(extract_features), filter(None), list)
    first_innings = chain(*[x[0] for x in data])
    second_innings = chain(*[x[1] for x in data])

    df1 = fit_model(first_innings, 1)
    df2 = fit_model(second_innings, 2)

    df1['innings'] = 1
    df2['innings'] = 2
    return pd.concat([df1, df2])

    
def win_probability_matrix_generator(params_1_df: pd.DataFrame,
                                     params_2_df: pd.DataFrame,
                                     max_over: int,
                                     max_runs: int) -> Iterator[Tuple[Prediction, Prediction]]:

    def average(metric1, metric2):
        wts_a = [1.0, 0.75, 0.5, 0.5, 0.25, 0.0]
        wts_b = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0]
        return metric1 * wts_a[ball - 1] + metric2 * wts_b[ball - 1]

    # generate the win probability for the entire state-space
    wickets = range(0, 11)
    runs = range(0, max_runs + 1)
    overs = range(1, max_over + 1)

    # we want to do a weighted average of the predictions

    params_1 = {row.overs: Parameter(row.overs, row.intercept, row.log_rr_param, row.log_wickets_param,
                                     row.mu, row.error_std) for row in params_1_df.itertuples()}

    params_2 = {row.overs: Parameter(row.overs, row.intercept, row.log_rr_param, row.log_wickets_param,
                                     row.mu, row.error_std) for row in params_2_df.itertuples()}
    for (over, wicket, run) in product(overs, wickets, runs):
        for ball in range(1, 7):
            run_rate = run / (((over - 1) * 6 + ball) / 6)
            feature_1a = Feature(1, over, ball, run, 0, run_rate, wicket)
            prediction_1a = predict_state(feature_1a, max_over, params_1, params_2)
            if over < max_over:
                feature_1b = Feature(1, over + 1, ball, run, 0, run_rate, wicket)
                prediction_1b = predict_state(feature_1b, max_over, params_1, params_2)

                wp_1a = prediction_1a.win_probability
                score_1a = prediction_1a.score
                score_lo_1a = prediction_1a.score_lo
                score_hi_1a = prediction_1a.score_hi

                wp_1b = prediction_1b.win_probability
                score_1b = prediction_1b.score
                score_lo_1b = prediction_1b.score_lo
                score_hi_1b = prediction_1b.score_hi

                score = average(score_1a, score_1b)
                score_lo = average(score_lo_1a, score_lo_1b)
                score_hi = average(score_hi_1a, score_hi_1b)
                wp = average(wp_1a, wp_1b)

                prediction_1 = Prediction(prediction_1a.Innings,
                                          prediction_1a.Over,
                                          prediction_1a.Ball,
                                          prediction_1a.Runs,
                                          prediction_1a.Target,
                                          prediction_1a.RunRate,
                                          prediction_1a.Wickets,
                                          wp, score, score_lo, score_hi)
            else:
                score_1a = prediction_1a.score
                score_lo_1a = prediction_1a.score_lo
                score_hi_1a = prediction_1a.score_hi

                # This is the score we would expect if we had 6 more balls.
                delta_score = score_1a - run
                delta_score_lo = score_lo_1a - run
                delta_score_hi = score_hi_1a - run

                score = int(run + (6 - ball) / 6.0 * delta_score)
                score_hi = int(run + (6 - ball) / 6.0 * delta_score_hi)
                score_lo = int(run + (6 - ball) / 6.0 * delta_score_lo)

                # obtain the wp for the predicted score now.
                params_predict = params_2[1]
                intercept = params_predict.intercept
                log_rr_param = params_predict.run_rate
                log_wickets_param = params_predict.wicket
                prediction = np.log(score / max_over + EPSILON_RUNS)
                reply = intercept + log_rr_param * prediction + log_wickets_param * np.log(10 + EPSILON_WICKETS)
                wp = 1 - np.exp(reply) / (1 + np.exp(reply))
                prediction_1 = Prediction(prediction_1a.Innings,
                                          prediction_1a.Over,
                                          prediction_1a.Ball,
                                          prediction_1a.Runs,
                                          prediction_1a.Target,
                                          prediction_1a.RunRate,
                                          prediction_1a.Wickets,
                                          wp, score, score_lo, score_hi)

            # Now run is the remaining runs to be scored.
            fractional_over = ((over - 1) * 6 + ball) / 6
            remaining_fractional_over = max_over - fractional_over
            required_run_rate = run / remaining_fractional_over if remaining_fractional_over > 0 else (0 if run == 0 else 36)
            feature_2a = Feature(2, over, ball, run, 0, required_run_rate, wicket)
            prediction_2a = predict_state(feature_2a, max_over, params_1, params_2)

            if over < max_over:
                feature_2b = Feature(2, over + 1, ball, run, 0, run / remaining_fractional_over, wicket)
                prediction_2b = predict_state(feature_2b, max_over, params_1, params_2)

                wp_2a = prediction_2a.win_probability
                wp_2b = prediction_2b.win_probability
                wp = average(wp_2a, wp_2b)

                prediction_2 = Prediction(prediction_2a.Innings,
                                          prediction_2a.Over,
                                          prediction_2a.Ball,
                                          prediction_2a.Runs,
                                          prediction_2a.Target,
                                          prediction_2a.RunRate,
                                          prediction_2a.Wickets,
                                          wp,
                                          prediction_2a.score,
                                          prediction_2a.score_lo,
                                          prediction_2a.score_hi)
            else:
                # the actual WP might be slightly higher because score 4 of 2 balls (RR=12) is probably easier than
                # scoring 12 of 6 balls.
                prediction_2 = prediction_2a
            yield prediction_1, prediction_2


def win_probability_matrix_generator_cricmetric_format(prediction: Tuple[Prediction, Prediction], is_odi: bool) -> Tuple[Text, Text]:
    """Returns the text form for the Cricmetric format.

    1st Innings: Balls, Wickets, Runs, WP, MID, LO, HI
    2nd Innings: Remaining_Balls, Remaining_Wickets, Remaining_Runs, WP

    :param prediction: The predictions for the given state
    :param is_odi: True if we are predicting ODI
    :return: tab separated text for each innings
    """

    def format(pred):
        innings, over, ball, runs, _, _, wickets, wp, mid, lo, hi = pred
        total_balls = (over - 1) * 6 + ball
        if innings == 2:
            remaining_runs = runs
            remaining_wickets = 10 - wickets
            remaining_balls = (300 if is_odi else 120) - total_balls
            wp_2nd_innings = 1 - wp
            return "\t".join(map(str, [remaining_balls, remaining_wickets, remaining_runs, wp_2nd_innings, mid, lo, hi]))
        return "\t".join(map(str, [total_balls, wickets, runs, wp, mid, lo, hi]))

    prediction_1, prediction_2 = prediction
    return format(prediction_1), format(prediction_2)


# Predictions & Figures

def predict_game(scorecard: pd.DataFrame, max_over: int,
                 params_1_df: pd.DataFrame, params_2_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Predicts the whole game from the given ball by ball data

    :param scorecard: ball by ball data frame
    :param max_over:  maximum overs in the game
    :param params_1_df: parameters for the first innings
    :param params_2_df: parameters for the second innings
    :return: A tuple of dataframes consisting of predictions for each innings.
    """

    params_1 = {row.overs: Parameter(row.overs, row.intercept, row.log_rr_param, row.log_wickets_param,
                                     row.mu, row.error_std) for row in params_1_df.itertuples()}

    params_2 = {row.overs: Parameter(row.overs, row.intercept, row.log_rr_param, row.log_wickets_param,
                                     row.mu, row.error_std) for row in params_2_df.itertuples()}

    states_1 = create_feature_vector(scorecard, innings=1, max_over=max_over)

    states_2 = create_feature_vector(scorecard, innings=2, max_over=max_over)

    predictions_1 = [predict_state(state, max_over, params_1, params_2) for state in states_1]
    predictions_2 = [predict_state(state, max_over, params_1, params_2) for state in states_2]

    df_1 = pd.DataFrame(predictions_1, columns=Prediction._fields)
    df_2 = pd.DataFrame(predictions_2, columns=Prediction._fields)

    df_1['overs_balls'] = df_1['Over'] - 1 + df_1['Ball'].apply(lambda x: min(x, 6.0)) / 6.0
    df_2['overs_balls'] = df_2['Over'] - 1 + df_2['Ball'].apply(lambda x: min(x, 6.0)) / 6.0

    return df_1, df_2


def plot_game(first_innings: pd.DataFrame, second_innings: pd.DataFrame, match_summary: pd.DataFrame,
              max_over: int) -> plt.Figure:
    """Visualizes the progress of the game.

    :param first_innings: Data frame with predictions for the 1st innings
    :param second_innings: Data frame with predictions for the 2nd innings
    :param match_summary: match summary data frame
    :param max_over: max overs in the game
    :return: A plt.Figure handle with the game visualized.
    """

    fid = first_innings.copy()
    sid = second_innings.copy()
    bbox_props = dict(boxstyle='round', fc='w', ec='0.5', alpha=0.9)

    def set_title(axis):
        link = 'http://www.espncricinfo.com/series/18902/statistics/{0}'.format(match_summary.Game_Id.values[0])
        axis.set_title('{0} v {1} @ {2} on {3}({4})'.format(team1, team2, venue, day, link))

    def worm_chart(idf, axis, color, ls):
        axis.plot(idf.overs1, idf.Runs, color=color, ls=ls)
        fow = idf[idf.foW > 0]
        axis.plot(fow.overs1, fow.Runs, 'o', markersize=10, color=color)

    def annotate_worm_chart(idf, axis, max_wickets=5):
        fow = idf[idf.foW > 0]
        for row in fow.itertuples():
            run = row.Runs
            wickets = row.Wickets
            if wickets <= max_wickets:
                axis.text(row.overs1, row.Runs + 10, '{0}-{1}'.format(int(run), int(wickets)))

    def predicted_score_chart(idf, axis, color, ls):
        axis.plot(idf.overs1, idf.score, color=color, ls=ls)
        axis.fill_between(idf.overs1, idf.score_lo, idf.score_hi, color=color, alpha=0.2)

    def box_score(idf, team, axis, color, factor):
        score = '{2}: {0} - {1}'.format(int(idf.Runs.max()), int(idf.Wickets.max()), team)
        axis.text(max_over * factor / 2, 50, score, ha='center', va='center', size=20, color=color, bbox=bbox_props)

    def format_xaxis(axis):
        axis.axvline(max_over, color='k')
        axis.set_xlim((0, 2 * max_over))
        axis.axvline(max_over / 2.0, color='k', linestyle='--')
        axis.axvline(max_over * 3.0 / 2.0, color='k', linestyle='--')
        axis.set_xticks(np.arange(0, 2 * max_over + 1, 4.0 if max_over == 20 else 5.0))
        if max_over == 50:
            axis.set_xticklabels([str(x) for x in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                                   5, 10, 15, 20, 25, 30, 35, 40, 45, 50)])
        else:
            axis.set_xticklabels([str(x) for x in (0, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20)])

    def format_yaxis(idf, axis):
        if max_over == 20:
            milestones = [50, 100, 150, 200, 250]
        else:
            milestones = [50, 100, 150, 200, 250, 300, 350, 400]

        axis.set_yticks(milestones)
        axis.set_yticklabels([str(x) for x in milestones])
        axis.set_ylim((0, idf.score_hi.max()))

    sid.rename(columns={'overs_balls': 'overs'}, inplace=True)
    fid.rename(columns={'overs_balls': 'overs'}, inplace=True)

    fid['overs1'] = fid.overs
    sid['overs1'] = sid.overs + max_over

    fid['foW'] = fid.Wickets - fid.Wickets.shift(1)
    sid['foW'] = sid.Wickets - sid.Wickets.shift(1)

    team1 = match_summary['1st_Innings'].values[0]
    team2 = match_summary['2nd_Innings'].values[0]
    venue = match_summary.Venue.values[0].split(',')[-1]
    day = match_summary.Match_Day.values[0].split('-')[0]

    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(211)

    worm_chart(fid, ax, 'r', '-')
    predicted_score_chart(fid, ax, 'r', '--')
    annotate_worm_chart(fid, ax)
    box_score(fid, team1, ax, 'r', 1)

    if len(sid) > 0:
        worm_chart(sid, ax, 'b', '-')
        plt.plot(sid.overs1, sid.Target, 'r', '--')
        annotate_worm_chart(sid, ax)
        box_score(sid, team2, ax, 'b', 3)

    format_xaxis(ax)
    format_yaxis(fid, ax)
    set_title(ax)

    ax2 = fig.add_subplot(212)

    ax2.plot(fid.overs1, fid.win_probability, color='r')
    if len(sid) > 0:
        ax2.plot(sid.overs1, sid.win_probability, color='r')

    format_xaxis(ax2)
    ax2.axhline(0.5, color='k', ls='--')
    ax2.text(max_over, 0.25, '{0} Win Probability'.format(team1), ha='center', va='center',
             size=20, color='red', bbox=bbox_props)
    ax2.set_ylim((0, 1.0))

    return fig
