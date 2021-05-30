""""Fetches the ball by ball details for a game."""
__author__ = 'kaushik'

from bs4 import BeautifulSoup
import datetime
import gcp
import pandas as pd
import requests
import warnings
from ball_by_ball import BallByBall
from match_details import MatchDetails

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
    for innings in (1, 2, 3, 4):
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


def get_skip_over(game_ids, bbb_data, ms_data):
    """Returns the set of games to download."""
    game_ids_from_ball_by_ball_data = set(bbb_data.Game_Id.unique())
    game_ids_from_match_summary_data = set(ms_data.Game_Id.unique())
    finished_ids = game_ids_from_ball_by_ball_data.intersection(game_ids_from_match_summary_data)
    skip_over_set = set([int(ids) for ids in finished_ids])
    game_ids_set = set(game_ids)
    ret = game_ids_set - skip_over_set
    print('Will fetch: {0} games'.format(len(ret)))
    return ret


def fetch_from_cricinfo(game_ids, print_frequency=25):
    total_games_to_fetch = len(game_ids)
    ball_by_balls = []
    match_summaries = []
    for i, game_id in enumerate(game_ids):
        if i % print_frequency == 0:
            print('Fetching {0} of {1}'.format(i, total_games_to_fetch))
        try:
            bbb, md = get_match_details(game_id)
        except Exception:
            print('Failed to fetch:{0}'.format(game_id))
        else:
            bbb_df = pd.DataFrame([b.to_row() for b in bbb], columns=BallByBall.get_header())
            md_df = pd.DataFrame([md.to_row()], columns=MatchDetails.get_header())
            ball_by_balls.append(bbb_df)
            match_summaries.append(md_df)
    return pd.concat(ball_by_balls), pd.concat(match_summaries)


def _update(game_ids, location):
    ball_by_ball_old = gcp.download_data_frame('ball-by-ball', location)
    match_summary_old = gcp.download_data_frame('match-summary', location)

    games_ids_to_fetch = get_skip_over(game_ids, ball_by_ball_old, match_summary_old)
    ball_by_ball_new, match_summary_new = fetch_from_cricinfo(games_ids_to_fetch)
    ball_by_ball = ball_by_ball_old.append(ball_by_ball_new)
    match_summary = match_summary_old.append(match_summary_new)

    gcp.upload_data_frame('ball-by-ball', location, ball_by_ball)
    gcp.upload_data_frame('match-summary', location, match_summary)


def update_odi():
    """Fetches all the game Ids for ODIs.
    """
    game_ids = []
    year = 2010
    season_end = datetime.datetime.now().year
    print('Crawling game ids...')
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
    _update(game_ids, 'ODI')


def update_ipl():
    data = BeautifulSoup(requests.get(
        'http://stats.espncricinfo.com/ci/engine/records/team/match_results_season.html?id=117;type=trophy').text,
                         features='lxml')
    season_links = []
    print('Crawling game ids...')
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

    _update(game_ids, 'IPL')


def update_t20i():
    data = BeautifulSoup(requests.get(
         'http://stats.espncricinfo.com/ci/content/records/307852.html').text,
                         features='lxml')
    season_links = []
    print('Crawling game ids...')
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
    _update(game_ids, 'T20I')


if __name__ == '__main__':
    print('Updating IPL...')
    update_ipl()

    print('Updating T2OI...')
    update_t20i()

    print('Updating ODI...')
    update_odi()

