"""Async version of the scraper."""


import asyncio
import logging
from os import path
import pathlib
from typing import Text, List, Tuple, Optional

import aiofiles
import aiohttp
from aiohttp.helpers import next_whole_second
import nest_asyncio
import pandas as pd
import toolz
from bs4 import BeautifulSoup

from ball_by_ball import BallByBall
from match_details import MatchDetails


nest_asyncio.apply()


CRICINFO = 'http://site.web.api.espn.com/apis/site/v2/sports/cricket/8676/'
# class = 1: Tests, class = 2: ODIs, class = 3: T20I, class = 6: all T20s
MATCH_ID_FETCHER = 'https://stats.espncricinfo.com/ci/engine/records/team/match_results.html?class={0};id={1};type=year'.format

logging.basicConfig(filename='scraper_async.log', level=logging.DEBUG, 
                    filemode='w',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',style='%' )
logging.info('Started')


def create_ball_by_ball_link(match: int,page: int, innings: int) -> Text:
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


def create_summary_link(match: int) -> Text:
    """Creates the link for the game summaries.
    
    Args:
      match: The game id
    
    Returns:
      The link (str)
    """
    link_stub = CRICINFO + 'summary?contentorigin=espn&event={0}&lang=en&region=us&section=cricinfo'
    return link_stub.format(match)

async def fetch_with_retry(link, session, max_retry):
    retry_count = 1
    while retry_count <= max_retry:
        response = await session.get(link)
        if response.status < 400:
            break
        retry_count += 1
    response.raise_for_status()
    return response

async def get_page_count(game_id: int, innings: int, session: aiohttp.ClientSession) -> int:
    """Returns the number of pages in the commentary."""
    link = create_ball_by_ball_link(game_id, innings=innings, page=1)
    response = await fetch_with_retry(link, session, 500)
    data = (await response.json())['commentary']
    return data['pageCount']

async def get_commentary_from_page(game_id: int, innings: int, session: aiohttp.ClientSession, page: int) -> List[BallByBall]:
    """Returns the ball by ball commentary information in the page."""
    link = create_ball_by_ball_link(game_id, innings=innings, page=page)
    response = await fetch_with_retry(link, session, 500)
    data = (await response.json())['commentary']
    return [BallByBall(game_id, item) for item in data['items']]


async def get_ball_by_ball_for_innings(game_id: int, innings: int, session: aiohttp.ClientSession) -> pd.DataFrame:
    """Returns the ball by ball commentary for the entire innings."""
    page_count = await get_page_count(game_id, innings, session)
    ball_by_ball = []
    for page in range(1, page_count+1):
        ball_by_ball += await get_commentary_from_page(game_id, innings, session, page)
    return pd.DataFrame([b.to_row() for b in ball_by_ball], columns=BallByBall.get_header())

async def get_ball_by_ball(game_id: int, game_type: Text, session: aiohttp.ClientSession) -> pd.DataFrame:
    """Returns the ball by ball commentary for the eniter game."""
    dfs = []
    all_innings = (1, 2) if game_type in ('ODI', 'T20I', 'T20') else (1, 2, 3, 4)
    for innings in all_innings:
        df = await get_ball_by_ball_for_innings(game_id, innings, session)
        dfs.append(df)
    return pd.concat(dfs)

async def get_summary(game_id: int, session: aiohttp.ClientSession) -> pd.DataFrame:
    """Returns the match summary."""
    link = create_summary_link(game_id)
    response = await fetch_with_retry(link, session, 500)
    data = await response.json()
    match_details = MatchDetails(game_id, data)
    return pd.DataFrame([match_details.to_row()], columns=MatchDetails.get_header())

    
async def write(game_id: int, session: aiohttp.ClientSession, game_type: Text, year: int) -> Tuple[int, int, bool]:
    """Fetches data and writes to files in the given locations."""
    try:
        bbb_df = await get_ball_by_ball(game_id, game_type, session)
        md_df = await get_summary(game_id, session)
    except aiohttp.ClientResponseError as c:
        logging.error(f'{game_id} had HTTP errors: {c}')
        return game_id, year, False
    except KeyError as k:
        logging.error(f'{game_id} had key error: {k}')
        return game_id, year, False
    else:
        location = f'{game_type}/{year}'
        async with aiofiles.open(location + '/ball_by_ball.csv', "a") as f:
            csv_str = bbb_df.to_csv(index=False, header=False)
            await f.write(csv_str)
        async with aiofiles.open(location +'/match_summary.csv', "a") as f:
            csv_str = md_df.to_csv(index=False, header=False)
            await f.write(csv_str)
        logging.info(f'{game_id} sucessfully downloaded')
        return game_id, year, True

async def get_game_ids(year: int, game_format: int, session: aiohttp.ClientSession) -> List[int]:
    """Returns the game ids for the game format in the year.
    game_format = 1: Test, 2: ODI, 3: T20Is, 6: all T20s
    """
    assert game_format in (1, 2, 3, 6)
    game_type = None
    if game_format == 1:
        game_type = 'TEST'
    elif game_format == 2:
        game_type = 'ODI'
    elif game_format == 3:
        game_type = 'T20I'
    else:
        game_type = 'T20'
    href_fn = lambda soup: soup.find_all('a', class_='data-link')  #  finds all the links
    filter_fn = lambda row: game_type in row.text.upper()  #  filters for test matches
    map_fn = lambda row: int(row['href'].split('/')[-1].split('.')[0])  # fetches match id
    link = MATCH_ID_FETCHER(game_format, year)
    response = await session.get(link)
    response.raise_for_status()
    text = await response.text()
    soup = BeautifulSoup(text, features='html.parser')
    return toolz.pipe(soup,
                      href_fn,
                      toolz.curried.filter(filter_fn),
                      toolz.curried.map(map_fn),
                      list
                     )

async def process_years(years: List[int], game_format: int) -> List[Tuple[int, bool]]:
    """Fetches the game details for a list of years. Returns the status"""

    def create_file(gt, yr):
        if not pathlib.Path(f'{gt}/{yr}').exists():
            logging.debug(f'Creating folder {gt}/{yr}')
            pathlib.Path(f'{gt}/{yr}').mkdir()
        if not pathlib.Path(f'{gt}/{yr}/ball_by_ball.csv').exists():
            logging.debug(f'Creating file {gt}/{yr}/ball_by_ball.csv')
            with open(f'{gt}/{yr}/ball_by_ball.csv', "w") as f:
                f.write(",".join(BallByBall.get_header()))
                f.write("\n")
        if not pathlib.Path(f'{gt}/{yr}/match_summary.csv').exists():
            logging.debug(f'Creating file {gt}/{yr}/match_summary.csv')
            with open(f'{gt}/{yr}/match_summary.csv', "w") as f:
                f.write(",".join(MatchDetails.get_header()))
                f.write("\n") 
    
    def get_ignore(gt, yr):
        """Gets the match ids to ignore in the year."""
        df = pd.read_csv(f'{gt}/{yr}/match_summary.csv')
        return set(df.Game_Id)
        
    assert game_format in (1, 2, 3, 6)
    game_type = None
    if game_format == 1:
        game_type = 'TEST'
    elif game_format == 2:
        game_type = 'ODI'
    elif game_format == 3:
        game_type = 'T20I'
    else:
        game_type = 'T20'
    # Make the directories if they do not exist
    if not pathlib.Path(game_type).exists():
        logging.debug(f'Creating folder {game_type}')
        pathlib.Path(game_type).mkdir()
    out = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for year in years:
            create_file(game_type, year)
            ignore = get_ignore(game_type, year)
            game_ids = await get_game_ids(year, game_format, session)
            for game_id in game_ids:
                if game_id in ignore:
                    continue
                tasks.append(write(game_id, session, game_type, year))
        out += [(await asyncio.gather(*tasks))]
    return out[0]

def _download(game_format: int, years: List[int]) -> List[Tuple[int, int, bool]]:
    return asyncio.run(process_years(years, game_format))

def download(game_format: int, years: List[int], max_retries: int=5) -> None:
    """Downloads and writes the game details in the respective directory."""
    tries = 0
    while years:
        resp = _download(game_format, years)
        missing = [x for x in resp if not x[-1]]
        years = set([x[1] for x in missing])
        logging.debug(f'Retrying for missed data in {years}. Num missing: {len(missing)}' )
        tries += 1
        if tries > max_retries:
            break
    return resp


if __name__ == '__main__':
    years = range(2007, 2022)
    logging.info('Downloading Tests...')
    missed = download(1, years)
    logging.debug(f'Missed in Tests: {missed}')
    
    # logging.info('Downloading ODIs...')
    # missed = download(2, years)
    # logging.debug(f'Missed in ODIs: {missed}')

    # logging.info('Downloading T20Is...')
    # missed = download(3, years)
    # logging.debug(f'Missed in T20Is: {missed}')

    # logging.info('Downloading T20s...')
    # missed = download(6, years)
    # logging.debug(f'Missed in T20Is: {missed}')
    
    
    
