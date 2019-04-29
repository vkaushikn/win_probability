# uncompyle6 version 3.3.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 2.7.16 (default, Apr 12 2019, 15:32:52) 
# [GCC 4.2.1 Compatible Apple LLVM 10.0.0 (clang-1000.11.45.5)]
# Embedded file name: /Users/kaushik/win_probability/fetch_ball_by_ball_details.py
# Size of source mod 2**32: 10596 bytes
"""Fetches the ball by ball details for a game.
"""
__author__ = 'kaushik'

import requests

LINK_STUB = 'http://site.web.api.espn.com/apis/site/v2/sports/cricket/8676/playbyplay?contentorigin=espn&event={0}&page={1}&period={2}&section=cricinfo'
SUMMARY_STUB = 'http://site.web.api.espn.com/apis/site/v2/sports/cricket/8676/summary?contentorigin=espn&event={0}&lang=en&region=us&section=cricinfo'

def create_link(match, page, innings):
    """Creates the link to fetch.
    
    Args:
      match: The game id to fetch
      page: The page to fetch
      innings: The innings to fetch
    
    Returns:
      The link (str)
    """
    return LINK_STUB.format(match, page, innings)


def create_summary_link(match):
    """Creates the link for the game summaries.
    
    Args:
      match: The game id
    
    Returns:
      The link (str)
    """
    return SUMMARY_STUB.format(match)


class BallByBall(object):
    """Contains the information about ball by ball data.
    """

    def __init__(self, game_id):
        self.game_id = game_id
        self.over_number = None
        self.ball_number = None
        self.batsman = None
        self.other_batsman = None
        self.bowler = None
        self.fours = None
        self.sixes = None
        self.wicket = None
        self.batsman_score = None
        self.other_batsman_score = None
        self.completed = None
        self.innings = None
        self.runs = None
        self.ball_limit = None
        self.remaining_balls = None
        self.target = None
        self.runrate = None
        self.extras = None
        self.byes = None
        self.legByes = None
        self.wide = None
        self.noBall = None
        self.remaining_runs = None
        self.required_run_rate = None

    def add_from_json(self, item):
        self.over_number = item['over']['number']
        self.ball_number = item['over']['unique']
        self.batsman = item['batsman']['athlete']['fullName']
        self.other_batsman = item['otherBatsman']['athlete']['fullName']
        self.bowler = item['bowler']['athlete']['fullName']
        self.fours = True if item['playType']['id'] == '3' else False
        self.sixes = True if item['playType']['id'] == '4' else False
        self.wicket = item['dismissal']['dismissal']
        self.runs = item['scoreValue']
        self.batsman_score = item['batsman']['totalRuns']
        self.other_batsman_score = item['otherBatsman']['totalRuns']
        self.completed = item['over']['complete']
        self.innings = item['period']
        self.ball_limit = item['innings']['ballLimit']
        self.remaining_balls = item['innings']['remainingBalls']
        self.target = item['innings']['target']
        self.runrate = item['innings']['runRate']
        self.extras = item['over']['byes'] + item['over']['legByes'] + item['over']['noBall'] + item['over']['wide']
        self.byes = item['over']['byes']
        self.legByes = item['over']['legByes']
        self.wide = item['over']['wide']
        self.noBall = item['over']['noBall']
        self.remaining_runs = item['innings'].get('remainingRuns', 0)
        self.required_run_rate = item['innings'].get('requiredRunRate', 0.0)

    def to_row(self):
        return [
         self.game_id,
         self.innings,
         self.over_number,
         self.ball_number,
         self.runs,
         self.fours,
         self.sixes,
         self.wicket,
         self.batsman_score,
         self.other_batsman_score,
         self.batsman,
         self.other_batsman,
         self.bowler,
         self.completed,
         self.ball_limit,
         self.remaining_balls,
         self.target,
         self.runrate,
         self.extras,
         self.byes,
         self.legByes,
         self.wide,
         self.noBall,
         self.remaining_runs,
         self.required_run_rate]

    @staticmethod
    def get_header():
        return [
         'Game_Id',
         'Innings',
         'Over',
         'Ball',
         'Runs',
         'Fours',
         'Sixes',
         'Wicket',
         'Batsman_Score',
         'Other_Batsman_Score',
         'Batsman',
         'Other_Batsman',
         'Bowler',
         'OverCompleted',
         'Ball_Limit',
         'Remaining_Balls',
         'Target',
         'Run_Rate',
         'Extras',
         'Byes',
         'LegByes',
         'Wide',
         'NoBall',
         'Remaining_Runs',
         'Required_Run_Rate']


class MatchDetails:
    """Gets the match summary."""

    def __init__(self, game_id):
        self.game_id = game_id
        self.first_bat_team = None
        self.second_bat_team = None
        self.toss_team = None
        self.toss_choice = None
        self.winning_innings = None
        self.venue = None
        self.reduced_over_game = None
        self.neutral_site_game = None
        self.day_night = None
        self.matchday = None

    def add_from_json(self, data):
        self.first_bat_team = None
        self.second_bat_team = None
        self.toss_team = None
        self.toss_choice = None
        self.winning_innings = 0
        self.venue = data['gameInfo']['venue']['fullName']
        self.reduced_over_game = data['header']['competitions'][0]['reducedOvers']
        self.neutral_site_game = data['header']['competitions'][0]['neutralSite']
        self.day_night = self._set_day_night(data['notes'])
        self._set_game_details(data['header']['competitions'][0]['competitors'])
        self._set_toss_team(data['notes'])
        self._get_match_day(data['notes'])

    def _set_game_details(self, competitors):
        for competitor in competitors:
            team = competitor['team']['name']
            innings = competitor['order']
            winner = competitor['winner']
            if isinstance(winner, str):
                if winner in ('True', 'true'):
                    winner = True
                else:
                    winner = False
            if innings == 1:
                self.first_bat_team = team
                if winner:
                    self.winning_innings = 1
                else:
                    self.second_bat_team = team
                    if winner:
                        self.winning_innings = 2

    def _set_toss_team(self, notes):
        for note in notes:
            if note['type'] not in ('toss', ):
                continue
            toss = note['text']
            self.toss_team = toss.split(',')[0].rstrip(' ')
            self.toss_choice = 'bat' if 'bat' in toss else 'field'

    def _get_match_day(self, notes):
        for note in notes:
            if note['type'] == 'matchdays':
                self.matchday = note['text']

    def _set_day_night(self, notes):
        matchdays = None
        hoursofplay = None
        for note in notes:
            if note['type'] not in ('hoursofplay', 'matchdays'):
                continue
            if note['type'] == 'hoursofplay':
                hoursofplay = note['text']
            else:
                matchdays = note['text']

        if matchdays is not None:
            if 'D/N' in matchdays or 'day/night' in matchdays:
                return 'D/N'
            return 'UNKNOWN'
        hours = hoursofplay[:5]
        if '.' in hours:
            hh, _ = hours.split('.')
            if int(hh) < 12:
                return 'D'
            return 'D/N'
        else:
            if ':' in hours:
                hh, _ = hours.split(':')
                if int(hh) < 12:
                    return 'D'
                return 'D/N'
            return 'UNKNOWN'

    def to_row(self):
        return [
         self.game_id,
         self.first_bat_team,
         self.second_bat_team,
         self.toss_team,
         self.toss_choice,
         self.winning_innings,
         self.reduced_over_game,
         self.neutral_site_game,
         self.venue,
         self.day_night,
         self.matchday]

    @staticmethod
    def get_header():
        return [
         'Game_Id',
         '1st_Innings',
         '2nd_Innings',
         'Toss',
         'Toss_Choice',
         'Winning_Innings',
         'Reduced_Over',
         'Neutral_Site',
         'Venue',
         'Day_Night',
         'Match_Day']


def _get_ball_by_ball(items, game_id):
    ball_by_ball = []
    for item in items:
        try:
            this_item = BallByBall(game_id)
            this_item.add_from_json(item)
            ball_by_ball.append(this_item)
        except:
            pass

    return ball_by_ball


def _get_ball_by_ball_for_match(game_id):
    ball_by_ball = []
    for innings in (1, 2):
        link = create_link(match=game_id, page=1, innings=innings)
        data = requests.get(link).json()['commentary']
        num_pages = data['pageCount']
        page = 1
        while page <= num_pages + 1:
            ball_by_ball += _get_ball_by_ball(data['items'], game_id)
            page += 1
            link = create_link(match=game_id, page=page, innings=innings)
            data = requests.get(link).json()['commentary']

    return ball_by_ball


def _get_game_details(game_id):
    link = create_summary_link(game_id)
    data = requests.get(link).json()
    match_details = MatchDetails(game_id)
    match_details.add_from_json(data)
    return match_details


def get_match_details(game_id):
    """Gets the details for a game.
    
    Args:
      game_id: The game to fetch
    
    Returns:
      Tuple (BallByBallDetails, MatchDetails)
    """
    ball_by_ball = _get_ball_by_ball_for_match(game_id)
    match_details = _get_game_details(game_id)
    return (
     ball_by_ball, match_details)


def get_match_summary(game_id):
    """Gets the match summary for a game.
    
    Args:
        game_id: The game to fetch
    
    Returns:
        MatchDetails
    """
    match_details = _get_game_details(game_id)
    return match_details