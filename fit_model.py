"""Fits the Logistic Regression Model to the data."""
__author__ = 'kaushik'

import pandas as pd
from collections import namedtuple
from glob import glob
from itertools import chain
import numpy as np
import re
from sklearn.linear_model import LinearRegression, LogisticRegression
from toolz import groupby, pipe
from toolz.curried import map, filter
from typing import List, Tuple, Text, Optional, Set, Iterator
import warnings


Feature = namedtuple("Feature", ["Over", "Ball", "RunRate", "Wickets"])
Target = namedtuple("Target", ["Target"])
Feature_Target_Pair = Tuple[Feature, Target]
Parameter = namedtuple("Parameter", ["over", "intercept", 'run_rate', 'wicket', 'mu', 'sigma'])


def validate_inputs(ball_by_ball_data, innings, max_over, result):
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


def create_feature_vector(ball_by_ball_data: pd.DataFrame,
                          result: int, innings: int, max_over: int) ->List[Feature_Target_Pair]:
    """

    :param ball_by_ball_data: A dataframe with the ball by ball data
    :param result: Result of the game (1 = chasing team has won the game)
    :param innings: 1 or 2
    :param max_over: max overs (50 or 20 depending on the game)
    :return: A pandas dataframe with the results. The columns will be
    [Over, Ball, RunRate, Wickets, Target].

    # Features == [Over, Ball, RunRate, Wickets] if first innings
    # Target == [EndRunRate]
    # Features == [Over, Ball, TargetRunRate, Wickets] if second innings
    # Target == [Result]
    """

    def get_features_and_target_init():
        run_rate =  0 if innings == 1 else inn.Target.values[0] / max_over
        target_0 = Target(inn.cumRuns.max() / max_over) if innings == 1 else Target(result)
        return Feature(0, 0, run_rate, 0), target_0

    def get_features_and_target_from_state(state):
        over = state.Over
        ball =  state.BallN
        run_rate = state.Run_Rate if innings == 1 else state.Required_Run_Rate
        wickets = state.cumWickets

        if innings == 2 and over == max_over and ball == 6:
            run_rate = 36.0 if state.cumRuns < state.Target else run_rate

        feature = Feature(over, ball, run_rate, wickets)
        target = Target(inn.cumRuns.max() / max_over) if innings == 1 else Target(result)
        return feature, target

    validate_inputs(ball_by_ball_data, innings, max_over, result)

    inn = ball_by_ball_data[ball_by_ball_data.Innings == innings]
    if len(inn) == 0:
        warnings.warn('Empty data frame')
        return []

    inn['BallN'] = (inn.Ball.apply(lambda x: x - x//1)) * 100
    inn['BallN'] = inn.BallN.apply(np.round)
    inn['OverKey'] = inn.Over - 1 + inn.BallN / 6
    inn.sort_values(by=['Over', 'BallN'], inplace=True)
    inn['cumRuns'] = inn.Runs.cumsum()
    inn['cumWickets'] = inn.Wicket.cumsum()
    return [get_features_and_target_init()] + [get_features_and_target_from_state(state) for state in inn.itertuples()]

def fit_model(features_and_targets: List[Feature_Target_Pair], innings: int) -> pd.DataFrame:


    def fit_model_for_over(pair: Tuple[int, List[Feature_Target_Pair]]) -> Parameter:

        def feature_vector(feature_target_pair: Feature_Target_Pair) -> np.array:
            feature, _ = feature_target_pair
            log_run_rate = np.log(feature.RunRate + 0.001)
            log_wicket = np.log(feature.Wickets + 0.001)
            return np.array([log_run_rate, log_wicket])

        def target_vector(feature_target_pair: Feature_Target_Pair) -> np.array:
            _, target = feature_target_pair
            return np.array(np.log(target.Target)) if innings == 1 else np.array(target.Target)

        def fit_first_innings(X: np.array, y:np.array, over:int) -> Parameter:
            lr = LinearRegression()
            lr.fit(X, y)
            predictions = lr.predict(X)
            errors = predictions - y
            return Parameter(over, lr.intercept_, lr.coef_[0], lr.coef_[1], np.mean(errors), np.std(errors))

        def fit_second_innings(X: np.array, y:np.array, over: int) -> Parameter:
            lr = LogisticRegression()
            lr.fit(X, y)
            return Parameter(over, lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1], 0, 0)

        over_num, feature_target_pair_list = pair
        features = pipe(feature_target_pair_list, map(feature_vector), list, np.array)
        targets = pipe(feature_target_pair_list, map(target_vector), list, np.array)
        return fit_first_innings(features, targets, over_num) if innings == 1 else \
            fit_second_innings(features, targets, over_num)

    def group_by_overs(fts: List[Feature_Target_Pair]) -> Iterator[Tuple[int, List[Feature_Target_Pair]]]:
        grouped = groupby(lambda x: x[0].Over, fts)
        return grouped.items()

    parameters = pipe(features_and_targets, group_by_overs, map(fit_model_for_over), list)
    return pd.DataFrame(parameters, columns=['overs', 'intercept',
                                             'log_rr_param', 'log_wickets_param', 'mu', 'error_std'])


def fit(location: Text, max_over: int, sample_percentage: float=1.0) -> None:
    """Coordinator function."""

    def get_game_ids(_) -> Set[Text]:
        ball_by_ball_match_string= "{0}/data/ball_by_ball/([0-9]*).csv".format(location)
        match_summary_match_string = "{0}/data/match_summary/([0-9]*).csv".format(location)

        ball_by_ball_file_names = glob('{0}/data/ball_by_ball/*.csv'.format(location))
        match_summary_file_names = glob('{0}/data/match_summary/*.csv'.format(location))
        game_ids_from_ball_by_ball_data = set(map(lambda x: re.match(ball_by_ball_match_string, x).groups()[0],
                                                  ball_by_ball_file_names))
        game_ids_from_match_summary_data = set(map(lambda x: re.match(match_summary_match_string, x).groups()[0],
                                                   match_summary_file_names))

        return game_ids_from_ball_by_ball_data.intersection(game_ids_from_match_summary_data)

    def extract_features_for_game(game_id : Text) -> List[Optional[Feature_Target_Pair]]:
        bd = pd.read_csv('{0}/data/ball_by_ball/{1}.csv'.format(location, game_id))
        md = pd.read_csv('{0}/data/match_summary/{1}.csv'.format(location, game_id))
        if md.Reduced_Over.values[0]:
            return []
        result = md.Winning_Innings.values[0] - 1

        if result not in [0, 1]:
            return []

        if np.random.random() > sample_percentage:
            return []
        f1 = create_feature_vector(bd, result, 1, max_over)
        f2 = create_feature_vector(bd, result, 2, max_over)

        if f1 and f2:
            return [f1, f2]
        return []

    data =  pipe(None, get_game_ids, map(extract_features_for_game), filter(None), list)
    print('finished creating data...')
    first_innings = chain(*[x[0] for x in data])
    second_innings = chain(*[x[1] for x in data])

    fit_model(first_innings, 1).to_csv('{0}/first_innings_parameters.csv'.format(location), index=False)
    fit_model(second_innings, 2).to_csv('{0}/second_innings_parameters.csv'.format(location), index=False)
    return