
"""Pulls the match summary details."""

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