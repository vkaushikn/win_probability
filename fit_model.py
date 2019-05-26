"""Fits the Logistic Regression Model to the data."""
__author__ = 'kaushik'

import pandas as pd
from collections import namedtuple
from glob import glob
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.linear_model import LinearRegression, LogisticRegression
from toolz import groupby, pipe
from toolz.curried import map, filter
from typing import List, Tuple, Text, Optional, Set, Iterator, NamedTuple, Dict
import warnings

pd.set_option('mode.chained_assignment', None)
EPSILON = 0.01

Feature = namedtuple("Feature", ["Over", "Ball", "Runs", "Target", "RunRate", "Wickets"])
Target = namedtuple("Target", ["Target"])
Feature_Target_Pair = Tuple[Feature, Target]
Parameter = namedtuple("Parameter", ["over", "intercept", 'run_rate', 'wicket', 'mu', 'sigma'])
Prediction = namedtuple("Prediction", ["Innings", "Over", "Ball", "Runs", "Target",
                                       "RunRate", "Wickets", "win_probability", "score", "score_lo", "score_hi"])


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


def get_features(state: NamedTuple, innings: int, max_over:int) -> Feature:
    over = state.Over
    ball =  state.BallN
    run_rate = state.Run_Rate if innings == 1 else state.Required_Run_Rate
    wickets = state.cumWickets
    runs = state.cumRuns
    target = state.Target

    if innings == 2 and over == max_over and ball == 6:
        run_rate = 36.0 if state.cumRuns < state.Target else run_rate

    feature = Feature(over, ball, runs, target, run_rate, wickets)
    return feature


def create_feature_vector(ball_by_ball_data: pd.DataFrame, innings: int, max_over: int) ->List[Feature]:

    validate_inputs(ball_by_ball_data, innings, max_over, 1)
    inn = get_innings_from_scorecard(ball_by_ball_data, innings)
    if len(inn) == 0:
        warnings.warn('Empty data frame')
        return []
    return [get_features(state, innings, max_over) for state in inn.itertuples()]

def create_feature_and_target_vector(ball_by_ball_data: pd.DataFrame,
                          result: int, innings: int, max_over: int) ->List[Feature_Target_Pair]:

    def get_features_and_target_from_state(state: NamedTuple):
        feature = get_features(state, innings, max_over)
        # target = Target(inn.cumRuns.max() / max_over) if innings == 1 else Target(result)
        target = Target(inn.tail(1).Run_Rate.values[0]) if innings == 1 else Target(result)
        return feature, target

    validate_inputs(ball_by_ball_data, innings, max_over, result)
    inn = get_innings_from_scorecard(ball_by_ball_data, innings)
    inn = inn[inn.OverCompleted]
    if len(inn) == 0:
        warnings.warn('Empty data frame')
        return []
    return [get_features_and_target_from_state(state) for state in inn.itertuples()]


def get_innings_from_scorecard(ball_by_ball_data, innings):
    inn = ball_by_ball_data[ball_by_ball_data.Innings == innings].copy(deep=True)
    inn['BallN'] = (inn.Ball.apply(lambda x: x - x // 1)) * 100
    inn['BallN'] = inn.BallN.apply(np.round)
    inn['OverKey'] = inn.Over - 1 + inn.BallN / 6
    inn.sort_values(by=['Over', 'BallN'], inplace=True)
    inn['cumRuns'] = inn.Runs.cumsum()
    inn['cumWickets'] = inn.Wicket.cumsum()
    return inn


def fit_model(features_and_targets: List[Feature_Target_Pair], innings: int) -> pd.DataFrame:


    def fit_model_for_over(pair: Tuple[int, List[Feature_Target_Pair]]) -> Parameter:

        def feature_vector(feature_target_pair: Feature_Target_Pair) -> np.array:
            feature, _ = feature_target_pair
            log_run_rate = np.log(feature.RunRate + EPSILON)
            log_wicket = np.log(feature.Wickets + EPSILON)
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
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(X, y)
            return Parameter(over, lr.intercept_[0], lr.coef_[0][0], lr.coef_[0][1], 0, 0)

        over_num, feature_target_pair_list = pair
        features = np.array([feature_vector(pair) for pair in feature_target_pair_list])
        targets = np.array([target_vector(pair) for pair in feature_target_pair_list])
        return fit_first_innings(features, targets, over_num) if innings == 1 else \
            fit_second_innings(features, targets, over_num)

    def group_by_overs(fts: List[Feature_Target_Pair]) -> Iterator[Tuple[int, List[Feature_Target_Pair]]]:
        grouped = groupby(lambda x: x[0].Over, fts)
        return grouped.items()

    parameters = [fit_model_for_over(pair) for pair in group_by_overs(features_and_targets)]
    return pd.DataFrame(parameters, columns=['overs', 'intercept',
                                             'log_rr_param', 'log_wickets_param', 'mu', 'error_std'])

def predict_state(state: Feature, max_over:int, params_1: Dict[int, Parameter],
                  params_2: Dict[int, Parameter], innings: int) -> Prediction:

    over = state.Over

    run_rate, wickets = state.RunRate, state.Wickets
    log_wickets = np.log(wickets + EPSILON)
    log_rr = np.log(run_rate + EPSILON)

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
        reply = intercept + log_rr_param * (prediction + EPSILON) + log_wickets_param * np.log(EPSILON)
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
        predicted_score = 'n/a'
        predicted_score_lo = 'n/a'
        predicted_score_hi = 'n/a'
        return Prediction(2, state.Over, state.Ball, state.Runs, state.Target,
                          state.RunRate, state.Wickets,
                          1 - win_probability, predicted_score, predicted_score_lo, predicted_score_hi)


def predict_game(scorecard: pd.DataFrame, max_over: int, params_1_df: pd.DataFrame, params_2_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    params_1 = {row.overs: Parameter(row.overs, row.intercept, row.log_rr_param, row.log_wickets_param,
                                     row.mu, row.error_std) for row in params_1_df.itertuples()}

    params_2 = {row.overs: Parameter(row.overs, row.intercept, row.log_rr_param, row.log_wickets_param,
                                     row.mu, row.error_std) for row in params_2_df.itertuples()}

    states_1 = create_feature_vector(scorecard, innings=1, max_over=max_over)

    states_2 = create_feature_vector(scorecard, innings=2, max_over=max_over)

    predictions_1 = [predict_state(state, max_over, params_1, params_2, 1) for state in states_1]
    predictions_2 = [predict_state(state, max_over, params_1, params_2, 2) for state in states_2]

    df_1 = pd.DataFrame(predictions_1, columns= Prediction._fields)
    df_2 = pd.DataFrame(predictions_2, columns= Prediction._fields)

    df_1['overs_balls'] = df_1['Over'] - 1 + df_1['Ball'].apply(lambda x: min(x, 6.0)) / 6.0
    df_2['overs_balls'] = df_2['Over'] - 1 + df_2['Ball'].apply(lambda x: min(x, 6.0)) / 6.0

    return df_1, df_2


def fit(location: Text, max_over: int, sample_percentage: float=1.0) -> pd.DataFrame:
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
        f1 = create_feature_and_target_vector(bd, result, 1, max_over)
        f2 = create_feature_and_target_vector(bd, result, 2, max_over)

        if f1 and f2:
            return [f1, f2]
        return []

    data =  pipe(None, get_game_ids, map(extract_features_for_game), filter(None), list)
    first_innings = chain(*[x[0] for x in data])
    second_innings = chain(*[x[1] for x in data])

    df1 = fit_model(first_innings, 1)
    df2 = fit_model(second_innings, 2)

    df1['innings'] = 1
    df2['innings'] = 2
    return pd.concat([df1, df2])


def plot_game(fid: pd.DataFrame, sid: pd.DataFrame, md: pd.DataFrame, max_over: int) -> plt.Figure:

    bbox_props = dict(boxstyle='round', fc='w', ec='0.5', alpha=0.9)

    def set_title(axis):
        axis.set_title('{0} v {1} @ {2} on {3}({4})'.
                 format(team1, team2, venue, day,
                        'http://www.espncricinfo.com/series/18902/statistics/{0}'.format(md.Game_Id.values[0])))

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

    def format_axis(axis):
        axis.axvline(max_over, color='k')
        axis.set_xlim((0, 2 * max_over))
        axis.axvline(max_over / 2.0, color='k', linestyle='--')
        axis.axvline(max_over * 3.0 / 2.0, color='k', linestyle='--')
        axis.set_xticks(np.arange(0, 2 * max_over + 1, 4.0 if max_over == 20 else 5.0))
        if max_over == 50:
            axis.set_xticklabels([str(x) for x in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                             50, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                             50)])
        else:
            axis.set_xticklabels([str(x) for x in (0, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20)])


    sid.rename(columns={'overs_balls': 'overs'}, inplace=True)
    fid.rename(columns={'overs_balls': 'overs'}, inplace=True)

    fid['overs1'] = fid.overs
    sid['overs1'] = sid.overs + max_over

    fid['foW'] = fid.Wickets - fid.Wickets.shift(1)
    sid['foW'] = sid.Wickets - sid.Wickets.shift(1)

    team1 = md['1st_Innings'].values[0]
    team2 = md['2nd_Innings'].values[0]
    venue = md.Venue.values[0].split(',')[-1]
    day = md.Match_Day.values[0].split('-')[0]

    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(211)

    worm_chart(fid, ax, 'r', '-')
    predicted_score_chart(fid, ax, 'r', '--')
    annotate_worm_chart(fid, ax)
    box_score(fid, team1, ax, 'r', 1
              )

    if len(sid) > 0:
        worm_chart(sid, ax, 'b', '-')
        plt.plt(sid.overs1, sid.Target, 'r', '--' )
        annotate_worm_chart(sid, ax)
        box_score(sid, team2, ax, 'b', 3)

    format_axis(ax)
    set_title(ax)
    ax.set_ylim((0, fid.score_hi.max()))



    ax = fig.add_subplot(212)

    ax.plot(fid.overs1, fid.win_probability, color='r')
    if len(sid) > 0:
        ax.plot(sid.overs1, sid.win_probability, color='r')

    format_axis(ax)
    ax.axhline(0.5, color='k', ls='--')
    ax.text(max_over, 0.25, '{0} Win Probability'.format(team1), ha='center', va='center',
            size=20, color='red', bbox=bbox_props)
    ax.set_ylim((0, 1.0))

    return fig