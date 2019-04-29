# uncompyle6 version 3.3.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 2.7.16 (default, Apr 12 2019, 15:32:52) 
# [GCC 4.2.1 Compatible Apple LLVM 10.0.0 (clang-1000.11.45.5)]
# Embedded file name: /Users/kaushik/win_probability/web_page.py
# Size of source mod 2**32: 1614 bytes
__author__ = 'kaushik'
from flask import Flask, render_template, send_file
from predictor import Predictor, predict_logistic, plot_progression
import pandas as pd, os
app = Flask(__name__)

@app.route('/predict_ipl/<int:game_id>')
def predict_ipl_game(game_id):
    predictor_instance = Predictor(20, '/Users/kaushik/win_probability/IPL/fit_1stInnings_LOGISTIC.csv', '/Users/kaushik/win_probability/IPL/fit_2ndInnings_LOGISTIC.csv', predict_logistic)
    try:
        bbb = pd.read_csv('/Users/kaushik/win_probability/IPL/data/ball_by_ball/{0}.csv'.format(game_id))
        md = pd.read_csv('/Users/kaushik/win_probability/IPL/data/match_summary/{0}.csv'.format(game_id))
        fid, sid = predictor_instance.predict_game(bbb)
        plot_progression(fid, sid, md, live=False,
          location='/Users/kaushik/win_probability/IPL/predictions/prediction_{0}.pdf'.format(game_id))
    except Exception as e:
        try:
            return 'Game Not FOUND:{0}'.format(game_id) + '\n' + e.__str__()
        finally:
            e = None
            del e

    return return_files(game_id)


@app.route('/prediction/prediction.pdf')
def return_files(game_id):
    return send_file('/Users/kaushik/win_probability/IPL/predictions/prediction_{0}.pdf'.format(game_id))


@app.route('/')
def index():
    return 'Usage: /predict_ipl/GAME_ID'