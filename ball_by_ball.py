"""Ball by Ball information."""


class BallByBall(object):
    """Contains the information about ball by ball data.
    """
    def __init__(self, game_id: int, data):
        self.game_id = game_id
        if 'name' not in data['team']:
            print(game_id)
        self.batting_team = data['team'].get('name', 'n/a')
        self.over_number = data['over']['number']
        self.ball_number = data['over']['unique']
        self.batsman = data['batsman']['athlete'].get('fullName', '')
        self.other_batsman = data['otherBatsman']['athlete'].get('fullName', '')
        self.bowler = data['bowler']['athlete'].get('fullName', '')
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
        self.runrate = data['innings'].get('runRate', 0.)
        self.extras = data['over']['byes'] + data['over']['legByes'] + data['over']['noBall'] + data['over']['wide']
        self.byes = data['over']['byes']
        self.legByes = data['over']['legByes']
        self.wide = data['over']['wide']
        self.noBall = data['over']['noBall']
        # Setting to zero if not populated - this data is not relevant for all innings.
        self.remaining_runs = data['innings'].get('remainingRuns', 0)
        self.required_run_rate = data['innings'].get('requiredRunRate', 0.0)
        self.day = data['innings'].get('day', 0.0)
        self.session = data['innings'].get('session', 0.0)
 

    def to_row(self):
        return [self.game_id, self.innings, self.batting_team, self.over_number, self.ball_number, self.runs, self.fours, self.sixes,
                self.wicket, self.batsman_score, self.other_batsman_score, self.batsman, self.other_batsman,
                self.bowler, self.completed, self.ball_limit, self.remaining_balls, self.target, self.runrate,
                self.extras, self.byes, self.legByes, self.wide, self.noBall, self.remaining_runs,
                self.required_run_rate, self.day, self.session]

    @staticmethod
    def get_header():
        return ['Game_Id', 'Innings', 'BattingTeam', 'Over', 'Ball', 'Runs', 'Fours', 'Sixes', 'Wicket', 'Batsman_Score',
                'Other_Batsman_Score', 'Batsman', 'Other_Batsman', 'Bowler', 'OverCompleted', 'Ball_Limit',
                'Remaining_Balls', 'Target', 'Run_Rate', 'Extras', 'Byes', 'LegByes', 'Wide', 'NoBall',
                'Remaining_Runs', 'Required_Run_Rate', 'Day', 'Session']