# uncompyle6 version 3.3.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 2.7.16 (default, Apr 12 2019, 15:32:52) 
# [GCC 4.2.1 Compatible Apple LLVM 10.0.0 (clang-1000.11.45.5)]
# Embedded file name: /Users/kaushik/win_probability/download_ipl_data.py
# Size of source mod 2**32: 3442 bytes
"""Downloads all the games from Cricinfo that have not been previously downloaded."""
__author__ = 'kaushik'
import fetch_ball_by_ball_details as fetch_details, pandas as pd, glob
from collections import defaultdict
import warnings
from bs4 import BeautifulSoup
import requests
BBB_HEADER = fetch_details.BallByBall.get_header()
MD_HEADER = fetch_details.MatchDetails.get_header()

def get_season_links():
    """Gets the Season Links from Cricinfo."""
    data = BeautifulSoup(requests.get('http://stats.espncricinfo.com/ci/engine/records/team/match_results_season.html?id=117;type=trophy').text,
      features='lxml')
    season_links = []
    for datum in data.find_all('table', class_='recordsTable'):
        links = datum.find_all('a')
        for link in links:
            season_links.append(link['href'])

    return season_links


def get_game_ids(season_links):
    """Fetches all the game Ids for IPL.
    """
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

    return game_ids


def get_skip_over(game_ids):
    """Returns the set of games to download."""
    bfiles = glob.glob('IPL/data/ball_by_ball/*.csv')
    mfiles = glob.glob('IPL/data/match_summary/*.csv')
    finished_ids = defaultdict(int)
    for bfile in bfiles:
        ids = bfile.split('.')[0].split('/')[-1]
        finished_ids[int(ids)] += 1

    for mfile in mfiles:
        ids = mfile.split('.')[0].split('/')[-1]
        finished_ids[int(ids)] += 1

    skip_over = [ids for ids, val in finished_ids.items() if val == 2]
    warnings.warn('Obtained :{0} games'.format(len(game_ids)))
    warnings.warn('Skipping over:{0} games'.format(len(skip_over)))
    game_ids_set = set(game_ids)
    skip_over_set = set(skip_over)
    ret = game_ids_set - skip_over_set
    warnings.warn('Will fetch: {0} games'.format(len(ret)))
    return ret


def fetch_and_write(game_ids, print_frequency=25):
    total_games_to_fetch = len(game_ids)
    for i, game_id in enumerate(game_ids):
        if i % print_frequency == 0:
            print('Fetching {0} of {1}'.format(i, total_games_to_fetch))
        try:
            bbb, md = fetch_details.get_match_details(game_id)
            bbb_df = pd.DataFrame([b.to_row() for b in bbb], columns=BBB_HEADER)
            md_df = pd.DataFrame([md.to_row()], columns=MD_HEADER)
            bbb_df.to_csv('IPL/data/ball_by_ball/{0}.csv'.format(game_id), index=None)
            md_df.to_csv('IPL/data/match_summary/{0}.csv'.format(game_id), index=None)
        except:
            warnings.warn('Failed to fetch:{0}'.format(game_id))


def download():
    season_links = get_season_links()
    game_ids = get_game_ids(season_links)
    games_ids_to_fetch = get_skip_over(game_ids)
    fetch_and_write(games_ids_to_fetch)