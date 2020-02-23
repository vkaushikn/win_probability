""""Fetches the ball by ball details for a game."""
__author__ = 'kaushik'

from bs4 import BeautifulSoup
import datetime
from functools import partial
from glob import glob
import pandas as pd
import re
import requests
from toolz import pipe
from toolz.curried import map, filter
from typing import Text, Optional
import warnings

CRICINFO = 'http://site.web.api.espn.com/apis/site/v2/sports/cricket/8676/'


def create_ball_by_ball_link(match, page, innings):
    """Creates the link to fetch.
    
    Args:
      match: The game id to fetch
      page: The page to fetch
      innings: The innings to fetch
    
    Returns:
      The link (str)
    """
    link_stub = CRICINFO + 'playbyplay?contentorigin=espn&event={0}&page={1}&period={2}&section=cricinfo'
    return link_stub.format(match, page, innings)


def create_summary_link(match):
    """Creates the link for the game summaries.
    
    Args:
      match: The game id
    
    Returns:
      The link (str)
    """
    link_stub = CRICINFO + 'summary?contentorigin=espn&event={0}&lang=en&region=us&section=cricinfo'
    return link_stub.format(match)


class BallByBall(object):
    """Contains the information about ball by ball data.
    """
    def __init__(self, game_id: int, data):
        self.game_id = game_id
        self.over_number = data['over']['number']
        self.ball_number = data['over']['unique']
        self.batsman = data['batsman']['athlete']['fullName']
        self.other_batsman = data['otherBatsman']['athlete']['fullName']
        self.bowler = data['bowler']['athlete']['fullName']
        self.fours = True if data['playType']['id'] == '3' else False
        self.sixes = True if data['playType']['id'] == '4' else False
        self.wicket = data['dismissal']['dismissal']
        self.runs = data['scoreValue']
        self.batsman_score = data['batsman']['totalRuns']
        self.other_batsman_score = data['otherBatsman']['totalRuns']
        self.completed = data['over']['complete']
        self.innings = data['period']
        self.ball_limit = data['innings']['ballLimit']
        self.remaining_balls = data['innings']['remainingBalls']
        self.target = data['innings']['target']
        self.runrate = data['innings']['runRate']
        self.extras = data['over']['byes'] + data['over']['legByes'] + data['over']['noBall'] + data['over']['wide']
        self.byes = data['over']['byes']
        self.legByes = data['over']['legByes']
        self.wide = data['over']['wide']
        self.noBall = data['over']['noBall']
        self.remaining_runs = data['innings'].get('remainingRuns', 0)
        self.required_run_rate = data['innings'].get('requiredRunRate', 0.0)

    def to_row(self):
        return [self.game_id, self.innings, self.over_number, self.ball_number, self.runs, self.fours, self.sixes,
                self.wicket, self.batsman_score, self.other_batsman_score, self.batsman, self.other_batsman,
                self.bowler, self.completed, self.ball_limit, self.remaining_balls, self.target, self.runrate,
                self.extras, self.byes, self.legByes, self.wide, self.noBall, self.remaining_runs,
                self.required_run_rate]

    @staticmethod
    def get_header():
        return ['Game_Id', 'Innings', 'Over', 'Ball', 'Runs', 'Fours', 'Sixes', 'Wicket', 'Batsman_Score',
                'Other_Batsman_Score', 'Batsman', 'Other_Batsman', 'Bowler', 'OverCompleted', 'Ball_Limit',
                'Remaining_Balls', 'Target', 'Run_Rate', 'Extras', 'Byes', 'LegByes', 'Wide', 'NoBall',
                'Remaining_Runs', 'Required_Run_Rate']


class MatchDetails(object):
    """Gets the match summary."""

    def __init__(self, game_id, data):
        self.game_id = game_id

        competitors = data['header']['competitions'][0]['competitors']
        self.first_bat_team = MatchDetails._get_first_bat_team(competitors)
        self.second_bat_team = MatchDetails._get_second_bat_team(competitors)
        self.winning_innings = MatchDetails._get_winning_innings(competitors)
        self.toss_team, self.toss_choice = MatchDetails._set_toss_team_and_choice(data['notes'])

        self.venue = data['gameInfo']['venue']['fullName']
        self.reduced_over_game = data['header']['competitions'][0]['reducedOvers']
        self.neutral_site_game = data['header']['competitions'][0]['neutralSite']

        self.day_night = MatchDetails._set_day_night(data['notes'])
        self.matchday = MatchDetails._get_match_day(data['notes'])

    def to_row(self):
        return [self.game_id, self.first_bat_team, self.second_bat_team, self.toss_team, self.toss_choice,
                self.winning_innings, self.reduced_over_game, self.neutral_site_game, self.venue, self.day_night,
                self.matchday]

    @staticmethod
    def _get_first_bat_team(competitors):
        for competitor in competitors:
            team = competitor['team']['name']
            innings = competitor['order']
            if innings == 1:
                return team

    @staticmethod
    def _get_second_bat_team(competitors):
        for competitor in competitors:
            team = competitor['team']['name']
            innings = competitor['order']
            if innings == 2:
                return team

    @staticmethod
    def _get_winning_innings(competitors):
        for competitor in competitors:
            winner = competitor['winner']
            if isinstance(winner, str):
                if winner in ('True', 'true'):
                    winner = True
                else:
                    winner = False
            innings = competitor['order']
            if innings == 1 and winner:
                return 1
            if innings == 2 and winner:
                return 2
        return 0

    @staticmethod
    def _set_toss_team_and_choice(notes):
        for note in notes:
            if note['type'] not in ('toss', ):
                continue
            toss = note['text']
            toss_team = toss.split(',')[0].rstrip(' ')
            toss_choice = 'bat' if 'bat' in toss else 'field'
            return toss_team, toss_choice

    @staticmethod
    def _get_match_day(notes):
        for note in notes:
            if note['type'] == 'matchdays':
                return note['text']

    @staticmethod
    def _set_day_night(notes):
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

    @staticmethod
    def get_header():
        return ['Game_Id', '1st_Innings', '2nd_Innings', 'Toss', 'Toss_Choice', 'Winning_Innings', 'Reduced_Over',
                'Neutral_Site', 'Venue', 'Day_Night', 'Match_Day']


def _get_ball_by_ball(items, game_id):
    ball_by_ball = []
    for item in items:
        try:
            this_item = BallByBall(game_id, item)
            ball_by_ball.append(this_item)
        except:
            pass
    return ball_by_ball


def _get_ball_by_ball_for_match(game_id):
    ball_by_ball = []
    for innings in (1, 2):
        link = create_ball_by_ball_link(match=game_id, page=1, innings=innings)
        data = requests.get(link).json()['commentary']
        num_pages = data['pageCount']
        page = 1
        while page <= num_pages + 1:
            ball_by_ball += _get_ball_by_ball(data['items'], game_id)
            page += 1
            link = create_ball_by_ball_link(match=game_id, page=page, innings=innings)
            data = requests.get(link).json()['commentary']
    return ball_by_ball


def _get_game_details(game_id):
    link = create_summary_link(game_id)
    data = requests.get(link).json()
    try:
        match_details = MatchDetails(game_id, data)
    except Exception:
        raise ValueError('Error fetching match details for id:{0}'.format(game_id))
    else:
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
    return ball_by_ball, match_details


def get_match_summary(game_id):
    """Gets the match summary for a game.
    
    Args:
        game_id: The game to fetch
    
    Returns:
        MatchDetails
    """
    match_details = _get_game_details(game_id)
    return match_details


# TODO(kaushik) Refactor matching into its own method - shared elsewhere
def get_skip_over(game_ids, location):
    """Returns the set of games to download."""

    def match(string: Text, pattern: Text) -> Optional[Text]:
        out = re.match(pattern, string)
        if out:
            return out.groups()[0]
        return None

    ball_by_ball_match_string = "{0}/data/ball_by_ball/([0-9]*).csv".format(location)
    match_summary_match_string = "{0}/data/match_summary/([0-9]*).csv".format(location)

    ball_by_ball_file_names = glob('{0}/data/ball_by_ball/*.csv'.format(location))
    match_summary_file_names = glob('{0}/data/match_summary/*.csv'.format(location))

    match_ball_by_ball = partial(match, pattern=ball_by_ball_match_string)
    game_ids_from_ball_by_ball_data = pipe(ball_by_ball_file_names, map(match_ball_by_ball), filter(None), set)

    match_match_summary = partial(match, pattern=match_summary_match_string)
    game_ids_from_match_summary_data = pipe(match_summary_file_names, map(match_match_summary), filter(None), set)

    finished_ids = game_ids_from_ball_by_ball_data.intersection(game_ids_from_match_summary_data)
    skip_over_set = set([int(ids) for ids in finished_ids])
    game_ids_set = set(game_ids)
    ret = game_ids_set - skip_over_set
    warnings.warn('Will fetch: {0} games'.format(len(ret)))
    return ret


def fetch_and_write(game_ids, location, print_frequency=25):
    total_games_to_fetch = len(game_ids)
    for i, game_id in enumerate(game_ids):
        if i % print_frequency == 0:
            print('Fetching {0} of {1}'.format(i, total_games_to_fetch))
        try:
            bbb, md = get_match_details(game_id)
        except Exception:
            warnings.warn('Failed to fetch:{0}'.format(game_id))
        else:
            bbb_df = pd.DataFrame([b.to_row() for b in bbb], columns=BallByBall.get_header())
            md_df = pd.DataFrame([md.to_row()], columns=MatchDetails.get_header())
            bbb_df.to_csv('{0}/data/ball_by_ball/{1}.csv'.format(location, game_id), index=None)
            md_df.to_csv('{0}/data/match_summary/{1}.csv'.format(location, game_id), index=None)


def _download(game_ids, location):
    games_ids_to_fetch = get_skip_over(game_ids, location)
    fetch_and_write(games_ids_to_fetch, location)


def download_odi():
    """Fetches all the game Ids for ODIs.
    """
    game_ids = []
    year = 2010
    season_end = datetime.datetime.now().year
    while year <= season_end:
        season_link =(
            'http://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class=2;id={0};type=year'.
             format(year))

        data = BeautifulSoup(requests.get(season_link).text, features='lxml')
        for datum in data.find_all('tr', class_='data1'):
            data_1 = datum.find_all('a')
            for datum_1 in data_1:
                if 'ODI' not in datum_1.text:
                    continue
                link = datum_1['href']
                game_id = link.split('/')[-1].split('.')[0]
                if game_id in game_ids:
                    warnings.warn('Repeated game_id: {0}'.format(game_id))
                game_ids.append(int(game_id))

        year += 1
    _download(game_ids, 'ODIs')


def download_ipl():
    data = BeautifulSoup(requests.get(
        'http://stats.espncricinfo.com/ci/engine/records/team/match_results_season.html?id=117;type=trophy').text,
                         features='lxml')
    season_links = []
    for datum in data.find_all('table', class_='recordsTable'):
        links = datum.find_all('a')
        for link in links:
            season_links.append(link['href'])

    game_ids = list()
    for season_link in season_links:
        data = BeautifulSoup(requests.get('http://stats.cricinfo.com' + season_link).text, features='lxml')
        for datum in data.find_all('tr', class_='data1'):
            data_1 = datum.find_all('a')
            for datum_1 in data_1:
                if datum_1.text != 'T20':
                    continue
                link = datum_1['href']
                game_id = link.split('/')[-1].split('.')[0]
                if game_id in game_ids:
                    warnings.warn('Repeated game_id: {0}'.format(game_id))
                game_ids.append(int(game_id))

    _download(game_ids, 'IPL')


def download_t20i():
    data = BeautifulSoup(requests.get(
         'http://stats.espncricinfo.com/ci/content/records/307852.html').text,
                         features='lxml')
    season_links = []
    for datum in data.find_all('table', class_='recordsTable'):
        links = datum.find_all('a')
        for link in links:
            season_links.append(link['href'])

    game_ids = list()
    for season_link in season_links:
        data = BeautifulSoup(requests.get('http://stats.cricinfo.com' + season_link).text, features='lxml')
        for datum in data.find_all('tr', class_='data1'):
            data_1 = datum.find_all('a')
            for datum_1 in data_1:
                if 'T20I' not in datum_1.text:
                    continue
                link = datum_1['href']
                game_id = link.split('/')[-1].split('.')[0]
                if game_id in game_ids:
                    warnings.warn('Repeated game_id: {0}'.format(game_id))
                game_ids.append(int(game_id))
    print('Downloading {0} games'.format(len(game_ids)))
    _download(game_ids, 'T20I')


if __name__ == '__main__':
    print('Downloading IPL...')
    download_ipl()

    print('Downloading T2OI...')
    download_t20i()

    print('Downloading ODI...')
    download_odi()

