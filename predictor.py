# uncompyle6 version 3.3.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 2.7.16 (default, Apr 12 2019, 15:32:52)
# [GCC 4.2.1 Compatible Apple LLVM 10.0.0 (clang-1000.11.45.5)]
# Embedded file name: /Users/kaushik/win_probability/predictor.py
# Size of source mod 2**32: 11594 bytes
"""Predicts the results of a single game."""
__author__ = 'kaushik'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Predictor(object):
    """Sets up the object that will help predict the game."""

    def __init__(self, max_over, first_innings_coeff_location, second_innings_coeff_location, predict_fn):
        """
        
        :param max_over:  Max overs
        :param first_innings_coeff_location: Data to load parameters from
        :param second_innings_coeff_location:  Data to load parameters from
        :param predict_fn: Function that predicts the game status
        :return:
        """
        self.max_over = max_over
        self.first_innings_coeff_location = first_innings_coeff_location
        self.second_innings_coeff_location = second_innings_coeff_location
        self.fit_df_1 = pd.read_csv(self.first_innings_coeff_location)
        self.fit_df_2 = pd.read_csv(self.second_innings_coeff_location)
        self.predict_fn = predict_fn

    def predict(self, innings, over, ball, run_rate, wickets, required_run_rate):
        """Returns the win probability of the given state (given the logistic regression model).
        
        Args:
            innings: Innings for which we want prediction
            over: Over
            ball: Ball
            run_rate: current run rate
            wickets: wickets fallen
            required_run_rate: required run rate if second innings
        
        Returns:
            Tuple of win probability, first innings score, upper bound of first innings score, lower bound of first
            innings score
        """
        if over > self.max_over and innings == 2:
            over == self.max_over
        if innings == 1:
            params = self.fit_df_1[self.fit_df_1.overs == over]
        else:
            params = self.fit_df_2[self.fit_df_2.overs == over]
        params_2 = self.fit_df_2[self.fit_df_2.overs == 1]
        return self.predict_fn(innings, run_rate, wickets, required_run_rate, self.max_over, params, params_2)

    def predict_game(self, ball_by_ball_df):
        """Returns the predictions for all the states in the input data frame."""
        columns = [
         'innings', 'overs_balls', 'over_num', 'ball_num', 'runs', 'WicketsLost', 'predicted_runs_m1sd',
         'predicted_runs', 'predicted_runs_p1sd', 'target', 'win_probability']
        df = ball_by_ball_df.copy()
        df['BallN'] = (df.Ball.apply(lambda x: x - x // 1)) * 100
        df['BallN'] = df.BallN.apply(np.round)
        df['OverKey'] = df.Over - 1 + df.BallN / 6
        first_inn = df[df.Innings == 1]
        first_inn['cumRuns'] = first_inn.Runs.cumsum()
        first_inn['cumWickets'] = first_inn.Wicket.cumsum()
        second_inn = df[df.Innings == 2]
        second_inn['cumRuns'] = second_inn.Runs.cumsum()
        second_inn['cumWickets'] = second_inn.Wicket.cumsum()

        rows = []
        for state in first_inn.itertuples():
            innings = 1
            ball = state.BallN
            over = state.Over
            runs = state.cumRuns
            wickets = state.cumWickets
            target = state.Target
            run_rate = state.Run_Rate
            required_run_rate = state.Required_Run_Rate
            win_probability, predicted_score_lo, predicted_score_hi, predicted_score = self.predict(innings, over, ball, run_rate, wickets, required_run_rate)
            rows.append([innings, over - 1 + ball / 6.0, over, ball, runs, wickets, predicted_score_lo, predicted_score,
             predicted_score_hi, target, win_probability])

        fid = pd.DataFrame(rows, columns=columns)
        fid.sort_values(by='overs_balls', inplace=True)

        rows = []
        for state in second_inn.itertuples():
            innings = 2
            ball = state.BallN
            over = state.Over
            runs = state.cumRuns
            wickets = state.cumWickets
            target = state.Target
            run_rate = state.Run_Rate
            required_run_rate = float(state.Required_Run_Rate)
            if over == self.max_over and ball > 5.1:
                if runs < target:
                    required_run_rate = (target - runs) * 36
            win_probability, predicted_score_lo, predicted_score_hi, predicted_score = self.predict(innings, over, ball, run_rate, wickets, required_run_rate)
            rows.append([innings, over -1 + ball / 6.0, over, ball, runs, wickets, predicted_score_lo, predicted_score,
             predicted_score_hi, target, win_probability])

        sid = pd.DataFrame(rows, columns=columns)
        sid.sort_values(by='overs_balls', inplace=True)
        return fid, sid


def plot_progression(fid, sid, md, max_over=20, game_format='IPL', live=False, location=None):
    """Plots the progression of the game.
    
    :param fid: First innings predictor output
    :param sid: Second innings predictor output
    :param md: match details
    :param fig: matplotlib fig
    :param max_over:  max over
    :param game_format: game format
    :param live: to print live probabilities
    :return: fig object
    """
    sid.rename(columns={'overs_balls': 'overs'}, inplace=True)
    fid.rename(columns={'overs_balls': 'overs'}, inplace=True)
    sid['overs1'] = sid.overs + max_over
    fig = plt.Figure(figsize=(20, 10))
    ax = fig.add_subplot(211)
    ax.plot(fid.overs, fid.runs, color='r', ls='-')
    ax.plot(fid.overs, fid.predicted_runs, color='r', ls='--')
    ax.fill_between(fid.overs, fid.predicted_runs_m1sd, fid.predicted_runs_p1sd, color='r', alpha=0.2)
    fid['foW'] = fid.WicketsLost - fid.WicketsLost.shift(1)
    fow = fid[fid.foW > 0]
    ax.plot(fow.overs, fow.runs, 'o', markersize=10, color='r')
    ax.plot(fow.overs, fow.predicted_runs, 'o', markersize=10, color='r')
    for row in fow.itertuples():
        run = row.runs
        wickets = row.WicketsLost
        if wickets <= 5:
            ax.text(row.overs, row.runs + 10, '{0}-{1}'.format(int(run), int(wickets)))

    team1 = md['1st_Innings'].values[0]
    team2 = md['2nd_Innings'].values[0]
    bbox_props = dict(boxstyle='round', fc='w', ec='0.5', alpha=0.9)
    box_score = '{2}: {0} - {1}'.format(int(fid.runs.max()), int(fid.WicketsLost.max()), team1)
    ax.text(max_over / 2, 50, box_score, ha='center', va='center', size=20, color='red', bbox=bbox_props)
    ax.axvline(max_over, color='k')
    if len(sid) > 0:
        ax.plot(sid.overs1, sid.runs, color='b', ls='-')
        ax.plot(sid.overs1, sid.target, color='r', ls='-.')
        sid['foW'] = sid.WicketsLost - sid.WicketsLost.shift(1)
        fow = sid[sid.foW > 0]
        ax.plot(fow.overs1, fow.runs, 'o', markersize=10, color='b')
        for row in fow.itertuples():
            run = row.runs
            wickets = row.WicketsLost
            if wickets <= 5:
                ax.text(row.overs + max_over, row.runs + 10, '{0}-{1}'.format(int(run), int(wickets)))

        box_score = '{2}: {0} - {1}'.format(int(sid.runs.max()), int(sid.WicketsLost.max()), team2)
        ax.text(max_over * 3 / 2.0, 50, box_score, ha='center', va='center', size=20, color='blue', bbox=bbox_props)
    ax.set_xlim((0, 2 * max_over))
    ax.set_ylim((0, fid.predicted_runs_p1sd.max()))
    ax.set_xticks(np.arange(0, 2 * max_over + 1, 4.0 if max_over == 20 else 5.0))
    if game_format == 'ODI':
        ax.set_xticklabels([str(x) for x in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                             50, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                             50)])
    else:
        ax.set_xticklabels([str(x) for x in (0, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20)])
    venue = md.Venue.values[0].split(',')[-1]
    day = md.Match_Day.values[0].split('-')[0]
    ax.set_title('{0} v {1} @ {2} on {3}({4})'.format(team1, team2, venue, day, 'http://www.espncricinfo.com/series/18902/statistics/{0}'.format(md.Game_Id.values[0])))
    ax = fig.add_subplot(212)
    ax.step(fid.overs, fid.win_probability, color='r')
    if len(sid) > 0:
        ax.step(sid.overs1, sid.win_probability, color='r')
    ax.axvline(max_over, color='k')
    ax.axvline(max_over / 2.0, color='k', linestyle='--')
    ax.axvline(max_over * 3.0 / 2.0, color='k', linestyle='--')
    ax.axhline(0.5, color='k')
    ax.set_xlim((0, 2 * max_over))
    ax.set_xticks(np.arange(0, 2 * max_over + 1, 4.0))
    if game_format == 'ODI':
        ax.set_xticklabels([str(x) for x in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                             50, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                                             50)])
    else:
        ax.set_xticklabels([str(x) for x in (0, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20)])
    ax.text(max_over, 0.25, '{0} Win Probability'.format(team1), ha='center', va='center', size=20, color='red', bbox=bbox_props)
    ax.set_ylim((0, 1.0))
    if live:
        wp = fid.tail(1)['win_probability'].values[0]
        if len(sid) > 0:
            wp = sid.tail(1)['win_probability'].values[0]
        ax.text(max_over, 0.1, box_score + '\n Win Probability:{0:.2f}'.format(wp * 100), ha='center',
          va='center',
          size=15,
          color='red')
    if location is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(location)
        pdf.savefig(fig)
        pdf.close()
        return
    else:
        return fig


def predict_logistic(innings, run_rate, wickets, required_run_rate, max_over, params, params_2):
    """Logistic Regression Based Predictor
    
    :param innings:
    :param run_rate:
    :param wickets:
    :param required_run_rate:
    :param max_over:
    :param params:
    :param params_2:
    :return:
    """
    intercept = params.intercept.values[0]
    log_rr_param = params.log_rr_param.values[0]
    log_wickets_param = params.log_wickets_param.values[0]
    std = params.error_std.values[0]
    log_wickets = np.log(wickets + 0.01)
    if innings == 1:
        log_rr = np.log(run_rate + 0.01)
        prediction = intercept + log_rr_param * log_rr + log_wickets_param * log_wickets
        intercept = params_2.intercept.values[0]
        log_rr_param = params_2.log_rr_param.values[0]
        log_wickets_param = params_2.log_wickets_param.values[0]
        reply = intercept + log_rr_param * prediction + log_wickets_param * np.log(0.01)
        win_probability = np.exp(reply) / (1 + np.exp(reply))
        predicted_score = np.exp(prediction) * max_over
        predicted_score_lo = np.exp(prediction - std) * max_over
        predicted_score_hi = np.exp(prediction + std) * max_over
        return (
         1 - win_probability, predicted_score_lo, predicted_score_hi, predicted_score)
    else:
        log_target_rr = np.log(required_run_rate + 0.01)
        prediction = intercept + log_rr_param * log_target_rr + log_wickets_param * log_wickets
        win_probability = np.exp(prediction) / (1 + np.exp(prediction))
        predicted_score = 'n/a'
        predicted_score_lo = 'n/a'
        predicted_score_hi = 'n/a'
        return (
         1 - win_probability, predicted_score_lo, predicted_score_hi, predicted_score)