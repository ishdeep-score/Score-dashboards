import re
import random
import streamlit as st
from mplsoccer import VerticalPitch,Pitch
import matplotlib.font_manager as font_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mplsoccer import FontManager
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
from highlight_text import ax_text, fig_text
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle
import glob,os
import matplotlib.image as mpimg
from mplsoccer import PyPizza, add_image, FontManager
import seaborn as sns
from matplotlib import colormaps
import joblib
from unidecode import unidecode
from scipy.ndimage import gaussian_filter


font_path = r'C:\Users\acer\Documents\GitHub\IndianCitizen\ScorePredict\Score Logos-20241022T100701Z-001\Score Logos\Sora_Font\Sora-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm_sora = FontManager()

team_dict = { 
        65: 'Barcelona',
        63: 'Atletico Madrid',
        52: 'Real Madrid',
        53: 'Atletic Club',
        839: 'Villarreal',
        54: 'Real Betis',
        64: 'Rayo Vallecano',
        51: 'Mallorca',
        68: 'Real Sociedad',
        62: 'Celta Vigo',
        131: 'Osasuna',
        67: 'Sevilla',
        2783: 'Girona',
        819: 'Getafe',
        70: 'Espanyol',
        825: 'Leganes',
        838: 'Las Palmas',
        55 : 'Valencia',
        60 : 'Deportivo Alaves',
        58: 'Real Valladolid',
        13: 'Arsenal',
        161: 'Wolves',
        24: 'Aston Villa',
        211: 'Brighton',
        30: 'Tottenham',
        167: 'Man City',
        14: 'Leicester',
        18: 'Southampton',
        183: 'Bournemouth',
        26: 'Liverpool',
        23: 'Newcastle',
        15: 'Chelsea',
        174: 'Nottingham Forest',
        29: 'West Ham',
        32: 'Man Utd',
        170: 'Fulham',
        189: 'Brentford',
        162: 'Crystal Palace',
        31: 'Everton',
        165: 'Ipswich',
        37: 'Bayern Munich',
        36: 'Bayer Leverkusen',
        45: 'Eintracht Frankfurt',
        219: 'Mainz 05',
        50: 'Freiburg',
        7614: 'RB Leipzig',
        33: 'Wolfsburg',
        134: 'Borussia M.Gladbach',
        41: 'VfB Stuttgart',
        44: 'Borussia Dortmund',
        1730: 'Augsburg',
        42: 'Werder Bremen',
        1211: 'Hoffenheim',
        796: 'Union Berlin',
        283: 'St. Pauli',
        1206: 'Holstein Kiel',
        4852: 'FC Heidenheim',
        109: 'Bochum',
        75 : 'Inter',
        276 : 'Napoli',
        300 : 'Atalanta',
        87 : 'Juventus',
        77 : 'Lazio',
        71 : 'Bologna',
        73 : 'Fiorentina',
        84 : 'Roma',
        80 : 'AC Milan',
        86 : 'Udinese',
        72 : 'Torino',
        278 : 'Genoa',
        1290 : 'Como',
        76 : 'Verona',
        78 : 'Cagliari',
        79 : 'Lecce',
        24341 : 'Parma Calcio',
        272 : 'Empoli',
        85 : 'Venezia',
        269 : 'Monza',
        304 : 'PSG',
        249 : 'Marseille',
        613 : 'Nice',
        248 : 'Monaco',
        607 : 'Lille',
        228 : 'Lyon',
        148 : 'Strasbourg',
        246 : 'Toulouse',
        309 : 'Lens',
        2332 : 'Brest',
        313 : 'Rennes',
        308 : 'Auxerre',
        614 : 'Angers',
        302 : 'Nantes',
        950 : 'Reims',
        217 : 'Le Havre',
        145 : 'Saint-Etienne',
        311 : 'Montpellier',
        299 : 'Benfica',
        129 : 'PSV',
        336 : 'Germany',
        340 : 'Portugal',
        338 : 'Spain',
        341 : 'France',
        342 : 'Poland',
        424 : 'Scotland',
        337 : 'Croatia',
        339 : 'Belgium',
        343 : 'Italy',
        325 : 'Israel',
        768 : 'Bosnia',
        335 : 'Netherlands',
        327 : 'Hungary',
        425 : 'Denmark',
        423 : 'Switzerland',
        771 : 'Serbia'
}
    
team_colors = {
    'Barcelona': '#A50044',
    'Atletico Madrid': '#CE3524',
    'Real Madrid': '#FCBF00',
    'Athletic Club': '#E0092C',
    'Villarreal': '#FFE667',
    'Real Betis': '#0BB363',
    'Rayo Vallecano': '#E53027',
    'Mallorca': '#E20613',
    'Real Sociedad': '#0067B1',
    'Celta Vigo': '#8AC3EE',
    'Osasuna': '#E60026',
    'Sevilla': '#D00027',
    'Girona': '#DA291C',
    'Getafe': '#003DA5',
    'Espanyol': '#00529F',
    'Leganes': '#005BAC',
    'Las Palmas': '#FEDD00',
    'Valencia': '#F18E00',
    'Deportivo Alaves': '#005BAC',
    'Real Valladolid': '#7A1E8B',
    'Arsenal': '#EF0107',
    'Wolves': '#FDB913',
    'Aston Villa': '#95BFE5',
    'Brighton': '#0057B8',
    'Tottenham': '#132257',
    'Man City': '#6CABDD',
    'Leicester': '#003090',
    'Southampton': '#D71920',
    'Bournemouth': '#DA291C',
    'Liverpool': '#C8102E',
    'Newcastle': "#F8F5F6",
    'Chelsea': '#034694',
    'Nottingham Forest': '#E53233',
    'West Ham': '#7A263A',
    'Man Utd': '#DA291C',
    'Fulham': "#FFFFFF",
    'Brentford': '#E30613',
    'Crystal Palace': '#1B458F',
    'Everton': '#003399',
    'Ipswich': '#005BAC',
    'Bayern Munich': '#DC052D',
    'Bayer Leverkusen': '#E30613',
    'Eintracht Frankfurt': "#FFFFFF",
    'Mainz 05': '#C8102E',
    'Freiburg': "#F9F4F4",
    'RB Leipzig': '#E4002B',
    'Wolfsburg': '#65B32E',
    'Borussia M.Gladbach': "#EEF3ED",
    'VfB Stuttgart': '#E30613',
    'Borussia Dortmund': '#FDE100',
    'Augsburg': '#C8102E',
    'Werder Bremen': '#1A9F3D',
    'Hoffenheim': '#005BAC',
    'Union Berlin': '#E30613',
    'St. Pauli': '#A45A2A',
    'Holstein Kiel': '#005BAC',
    'FC Heidenheim': '#E30613',
    'Bochum': '#005BAC',
    'Inter': '#1E2943',
    'Napoli': '#0082CA',
    'Atalanta': '#1C1C1C',
    'Juventus': "#F8F4F4",
    'Lazio': '#A8C6E5',
    'Bologna': '#D4001F',
    'Fiorentina': '#592C82',
    'Roma': '#8E1111',
    'AC Milan': '#FB090B',
    'Udinese': "#F2EBEB",
    'Torino': '#8B1B3A',
    'Genoa': '#C8102E',
    'Como': '#005BAC',
    'Verona': '#FCE500',
    'Cagliari': '#C8102E',
    'Lecce': '#FCE500',
    'Parma Calcio': '#FCE500',
    'Empoli': '#005BAC',
    'Venezia': "#F2ECEC",
    'Monza': '#E30613',
    'PSG': '#004170',
    'Marseille': '#009DDC',
    'Nice': '#E30613',
    'Monaco': '#ED1C24',
    'Lille': '#E30613',
    'Lyon': '#E30613',
    'Strasbourg': '#005BAC',
    'Toulouse': '#5F259F',
    'Lens': '#E30613',
    'Brest': '#E30613',
    'Rennes': '#E30613',
    'Auxerre': '#005BAC',
    'Angers': "#EEECEC",
    'Nantes': '#FCE500',
    'Reims': '#E30613',
    'Le Havre': '#005BAC',
    'Saint-Etienne': '#009639',
    'Montpellier': '#005BAC',
    'Benfica': '#E30613',
    'PSV': '#E30613',
    'Ajax': '#E30613',
    'Feyenoord': '#E30613',
    'Utrecht': '#E30613',
    'AZ-Alkmaar': '#E30613',
    'Twente': '#E30613',
    'Go Ahead Eagles': '#E30613',
    'FC Groningen': '#007A33',
    'Fortuna Sittard': '#FCE500',
    'Heracles': "#FAF7F7",
    'SC Heerenveen': '#005BAC',
    'NEC Nijmegen': '#E30613',
    'NAC Breda': '#FCE500',
    'PEC Zwolle': '#005BAC',
    'Sparta Rotterdam': '#E30613',
    'Willem II': '#E30613',
    'RKC Waalwijk': '#FCE500',
    'Almere City': '#E30613'
    }

@st.cache_data

def load_data(root_folder):
    csv_files = glob.glob(os.path.join(root_folder, "**", "*.csv"), recursive=True)
    dataframes = []
    for file in csv_files:
        try:
            dfi = pd.read_csv(file)
            dataframes.append(dfi)
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
    if dataframes:
        df = pd.concat(dataframes, ignore_index=True)
    else:
        df = pd.DataFrame()  # empty fallback
    return df, csv_files

def highlight_higher(val, all_vals):
    return 'color: green; font-weight: bold' if val == max(all_vals) else ''

def insert_ball_carries(events_df, min_carry_length, max_carry_length, min_carry_duration, max_carry_duration):
    events_out = pd.DataFrame()
    # match_events = events_df[events_df['match_id'] == match_id].reset_index()
    match_events = events_df.reset_index()
    match_carries = pd.DataFrame()

    for idx, match_event in match_events.iterrows():

        if idx < len(match_events) - 1:
            prev_evt_team = match_event['teamId']
            next_evt_idx = idx + 1
            init_next_evt = match_events.loc[next_evt_idx]
            take_ons = 0
            incorrect_next_evt = True

            while incorrect_next_evt:

                if next_evt_idx >= len(match_events):
                    break
                next_evt = match_events.iloc[next_evt_idx]

                if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                    take_ons += 1
                    incorrect_next_evt = True

                elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                      or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt['outcomeType'] == 'Unsuccessful')
                      or (next_evt['type'] == 'Foul')):
                    incorrect_next_evt = True

                else:
                    incorrect_next_evt = False

                next_evt_idx += 1

            # Apply some conditioning to determine whether carry criteria is satisfied
            same_team = prev_evt_team == next_evt['teamId']
            not_ball_touch = match_event['type'] != 'BallTouch'
            dx = 105*(match_event['endX'] - next_evt['x'])/100
            dy = 68*(match_event['endY'] - next_evt['y'])/100
            far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
            not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
            dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
            min_time = dt >= min_carry_duration
            same_phase = dt < max_carry_duration
            same_period = match_event['period'] == next_evt['period']

            valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase &same_period

            if valid_carry:
                carry = pd.DataFrame()
                prev = match_event
                nex = next_evt

                carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                        prev['minute'] * 60 + prev['second'])) / (2 * 60))
                carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                    (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                carry['teamId'] = nex['teamId']
                carry['x'] = prev['endX']
                carry['y'] = prev['endY']
                carry['expandedMinute'] = np.floor(((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                                                    (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                carry['period'] = nex['period']
                carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                carry['outcomeType'] = 'Successful'
                carry['qualifiers'] = carry.apply(lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                carry['isTouch'] = True
                carry['playerId'] = nex['playerId']
                carry['endX'] = nex['x']
                carry['endY'] = nex['y']
                carry['blockedX'] = np.nan
                carry['blockedY'] = np.nan
                carry['goalMouthZ'] = np.nan
                carry['goalMouthY'] = np.nan
                carry['isShot'] = np.nan
                carry['relatedEventId'] = nex['eventId']
                carry['relatedPlayerId'] = np.nan
                carry['isGoal'] = np.nan
                carry['cardType'] = np.nan
                carry['isOwnGoal'] = np.nan
                carry['type'] = 'Carry'
                carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2
                carry['playerName'] = nex['playerName']

                match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)

    match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
    match_events_and_carries = match_events_and_carries.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)

    # Rebuild events dataframe
    events_out = pd.concat([events_out, match_events_and_carries])

    return events_out

def cumulative_match_mins(events_df):
    events_out = pd.DataFrame()
    # Add cumulative time to events data, resetting for each unique match
    match_events = events_df.copy()
    match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
    # Add time increment to cumulative minutes based on period of game.
    for period in np.arange(1, match_events['period'].max() + 1, 1):
        if period > 1:
            t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                                   match_events[match_events['period'] == period]['cumulative_mins'].min()
        elif period == 1 or period == 5:
            t_delta = 0
        else:
            t_delta = 0
        match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
    # Rebuild events dataframe
    events_out = pd.concat([events_out, match_events])
    return events_out

def get_match_df(df, home_team, away_team,team_dict,team_colors):
    # Find matchId(s) where selected home_team was home ('h') AND away_team was away ('a')
    matching_matches = df[
        (df['teamName'] == home_team) & (df['h_a'] == 'h')
    ]['matchId'].unique()

    # Filter further: check that in the same matchId, the away team appears as 'a'
    valid_match_ids = []
    for match_id in matching_matches:
        away_team_rows = df[(df['matchId'] == match_id) & (df['teamName'] == away_team) & (df['h_a'] == 'a')]
        if not away_team_rows.empty:
            valid_match_ids.append(match_id)

    if not valid_match_ids:
        st.warning(f"No match found between {home_team} (home) and {away_team} (away).")
    else:
        # Assuming only 1 matchId between them per venue
        selected_match_id = valid_match_ids[0]

        # Filter df to keep only events from this match
        match_df = df[df['matchId'] == selected_match_id]

        match_df['period'] = match_df['period'].replace({'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3, 'SecondPeriodOfExtraTime': 4,
                                     'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16})
        match_df = cumulative_match_mins(match_df)
        match_df = insert_ball_carries(match_df, min_carry_length=15, max_carry_length=60, min_carry_duration=4, max_carry_duration=10)
        match_df = match_df.reset_index(drop=True)
        match_df['index'] = range(1, len(match_df) + 1)
        match_df = match_df[['index'] + [col for col in match_df.columns if col != 'index']]

        df_base  = match_df
        dfxT = df_base.copy()
        dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
        dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
        dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType']=='Successful')]


        xT = pd.read_csv('https://raw.githubusercontent.com/mckayjohns/youtube-videos/main/data/xT_Grid.csv', header=None) # use this if you don't have your own xT value Grid
        # xT = pd.read_csv("/content/xT_Grid.csv", header=None)    # use this if you have your own xT value Grid, then place your file path here
        xT = np.array(xT)
        xT_rows, xT_cols = xT.shape

        dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
        dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
        dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
        dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)

        dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
        dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)

        dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
        columns_to_drop = ['eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period', 'type', 'outcomeType', 'qualifiers', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'relatedEventId', 'relatedPlayerId', 'isGoal', 'cardType', 'isOwnGoal', 'cumulative_mins', 'Unnamed: 0', 'id', 'h_a', 'matchId', 'startDate', 'startTime', 'score', 'ftScore', 'htScore', 'etScore', 'venueName', 'maxMinute', 'playerName', 'shotBodyType', 'situation', 'shotSixYardBox', 'shotPenaltyArea', 'shotOboxTotal', 'shotOpenPlay', 'shotCounter', 'shotSetPiece', 'shotDirectCorner', 'shotOffTarget', 'shotOnPost', 'shotOnTarget', 'shotsTotal', 'shotBlocked', 'shotRightFoot', 'shotLeftFoot', 'shotHead', 'shotObp', 'goalSixYardBox', 'goalPenaltyArea', 'goalObox', 'goalOpenPlay', 'goalCounter', 'goalSetPiece', 'penaltyScored', 'goalOwn', 'goalNormal', 'goalRightFoot', 'goalLeftFoot', 'goalHead', 'goalObp', 'shortPassInaccurate', 'shortPassAccurate', 'passCorner', 'passCornerAccurate', 'passCornerInaccurate', 'passFreekick', 'passBack', 'passForward', 'passLeft', 'passRight', 'keyPassLong', 'keyPassShort', 'keyPassCross', 'keyPassCorner', 'keyPassThroughball', 'keyPassFreekick', 'keyPassThrowin', 'keyPassOther', 'assistCross', 'assistCorner', 'assistThroughball', 'assistFreekick', 'assistThrowin', 'assistOther', 'dribbleLost', 'dribbleWon', 'challengeLost', 'interceptionWon', 'clearanceHead', 'outfielderBlock', 'passCrossBlockedDefensive', 'outfielderBlockedPass', 'offsideGiven', 'offsideProvoked', 'foulGiven', 'foulCommitted', 'yellowCard', 'voidYellowCard', 'secondYellow', 'redCard', 'turnover', 'dispossessed', 'saveLowLeft', 'saveHighLeft', 'saveLowCentre', 'saveHighCentre', 'saveLowRight', 'saveHighRight', 'saveHands', 'saveFeet', 'saveObp', 'saveSixYardBox', 'savePenaltyArea', 'saveObox', 'keeperDivingSave', 'standingSave', 'closeMissHigh', 'closeMissHighLeft', 'closeMissHighRight', 'closeMissLeft', 'closeMissRight', 'shotOffTargetInsideBox', 'touches', 'assist', 'ballRecovery', 'clearanceEffective', 'clearanceTotal', 'clearanceOffTheLine', 'dribbleLastman', 'errorLeadsToGoal', 'errorLeadsToShot', 'intentionalAssist', 'interceptionAll', 'interceptionIntheBox', 'keeperClaimHighLost', 'keeperClaimHighWon', 'keeperClaimLost', 'keeperClaimWon', 'keeperOneToOneWon', 'parriedDanger', 'parriedSafe', 'collected', 'keeperPenaltySaved', 'keeperSaveInTheBox', 'keeperSaveTotal', 'keeperSmother', 'keeperSweeperLost', 'keeperMissed', 'passAccurate', 'passBackZoneInaccurate', 'passForwardZoneAccurate', 'passInaccurate', 'passAccuracy', 'cornerAwarded', 'passKey', 'passChipped', 'passCrossAccurate', 'passCrossInaccurate', 'passLongBallAccurate', 'passLongBallInaccurate', 'passThroughBallAccurate', 'passThroughBallInaccurate', 'passThroughBallInacurate', 'passFreekickAccurate', 'passFreekickInaccurate', 'penaltyConceded', 'penaltyMissed', 'penaltyWon', 'passRightFoot', 'passLeftFoot', 'passHead', 'sixYardBlock', 'tackleLastMan', 'tackleLost', 'tackleWon', 'cleanSheetGK', 'cleanSheetDL', 'cleanSheetDC', 'cleanSheetDR', 'cleanSheetDML', 'cleanSheetDMC', 'cleanSheetDMR', 'cleanSheetML', 'cleanSheetMC', 'cleanSheetMR', 'cleanSheetAML', 'cleanSheetAMC', 'cleanSheetAMR', 'cleanSheetFWL', 'cleanSheetFW', 'cleanSheetFWR', 'cleanSheetSub', 'goalConcededByTeamGK', 'goalConcededByTeamDL', 'goalConcededByTeamDC', 'goalConcededByTeamDR', 'goalConcededByTeamDML', 'goalConcededByTeamDMC', 'goalConcededByTeamDMR', 'goalConcededByTeamML', 'goalConcededByTeamMC', 'goalConcededByTeamMR', 'goalConcededByTeamAML', 'goalConcededByTeamAMC', 'goalConcededByTeamAMR', 'goalConcededByTeamFWL', 'goalConcededByTeamFW', 'goalConcededByTeamFWR', 'goalConcededByTeamSub', 'goalConcededOutsideBoxGoalkeeper', 'goalScoredByTeamGK', 'goalScoredByTeamDL', 'goalScoredByTeamDC', 'goalScoredByTeamDR', 'goalScoredByTeamDML', 'goalScoredByTeamDMC', 'goalScoredByTeamDMR', 'goalScoredByTeamML', 'goalScoredByTeamMC', 'goalScoredByTeamMR', 'goalScoredByTeamAML', 'goalScoredByTeamAMC', 'goalScoredByTeamAMR', 'goalScoredByTeamFWL', 'goalScoredByTeamFW', 'goalScoredByTeamFWR', 'goalScoredByTeamSub', 'aerialSuccess', 'duelAerialWon', 'duelAerialLost', 'offensiveDuel', 'defensiveDuel', 'bigChanceMissed', 'bigChanceScored', 'bigChanceCreated', 'overrun', 'successfulFinalThirdPasses', 'punches', 'penaltyShootoutScored', 'penaltyShootoutMissedOffTarget', 'penaltyShootoutSaved', 'penaltyShootoutSavedGK', 'penaltyShootoutConcededGK', 'throwIn', 'subOn', 'subOff', 'defensiveThird', 'midThird', 'finalThird', 'pos']
        dfxT.drop(columns=columns_to_drop, inplace=True)

        match_df = match_df.merge(dfxT, on='index', how='left')

        match_df['x'] = match_df['x']*1.05
        match_df['y'] = match_df['y']*0.68
        match_df['endX'] = match_df['endX']*1.05
        match_df['endY'] = match_df['endY']*0.68
        match_df['goalMouthY'] = match_df['goalMouthY']*0.68

        match_df['qualifiers'] = match_df['qualifiers'].astype(str)
        # Calculating passing distance, to find out progressive pass, this will just show the distance reduced by a pass, then will be able to filter passes which has reduced distance value more than 10yds as a progressive pass
        match_df['prog_pass'] = np.where((match_df['type'] == 'Pass'),
                                np.sqrt((105 - match_df['x'])**2 + (34 - match_df['y'])**2) - np.sqrt((105 - match_df['endX'])**2 + (34 - match_df['endY'])**2), 0)
        # Calculating carrying distance, to find out progressive carry, this will just show the distance reduced by a carry, then will be able to filter carries which has reduced distance value more than 10yds as a progressive carry
        match_df['prog_carry'] = np.where((match_df['type'] == 'Carry'),
                                    np.sqrt((105 - match_df['x'])**2 + (34 - match_df['y'])**2) - np.sqrt((105 - match_df['endX'])**2 + (34 - match_df['endY'])**2), 0)
        match_df['pass_or_carry_angle'] = np.degrees(np.arctan2(match_df['endY'] - match_df['y'], match_df['endX'] - match_df['x']))

        columns_to_drop2 = ['id']
        match_df.drop(columns=columns_to_drop2, inplace=True)

        match_df['period'] = match_df['period'].replace({1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime',
                                            5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'})

        match_df['teamName'] = match_df['teamId'].map(team_dict)
        match_df['teamColor'] = match_df['teamName'].map(team_colors)
        match_df = match_df.sort_values(by='index').reset_index(drop=True)

        match_df['event_time'] = match_df['minute'] * 60 + match_df['second']

        # Get first sub time per team
        first_sub_times = match_df[match_df['type'] == 'SubstitutionOn'] \
            .groupby('teamName')['event_time'].min().to_dict()

        # Set isFirstEleven default
        def is_first_eleven(row):
            team = row['teamName']
            t = row['event_time']
            return t < first_sub_times.get(team, float('inf'))

        match_df['isFirstEleven'] = match_df.apply(is_first_eleven, axis=1)

        # Force all SubstitutionOn players to False
        match_df.loc[match_df['type'] == 'SubstitutionOn', 'isFirstEleven'] = False

    return match_df

def get_team_names(df,team_dict,team_colors):
    df = df.sort_values(by='matchId')
    df['teamName'] = df['teamId'].map(team_dict)
    df['teamColor'] = df['teamName'].map(team_colors)
    team_names = list(team_dict.values())
    return df,df['teamName'].unique()

def plot_shotmap_understat_player(df_shots,team,league,teamcolor,player,season,situation,shotType):
    df1 = df_shots[(df_shots['h_team'] == team) & (df_shots['h_a'] == 'h')]
    df2 = df_shots[(df_shots['a_team'] == team) & (df_shots['h_a'] == 'a')]
    teamdf = pd.concat([df1, df2], ignore_index=True)
    df = teamdf[teamdf['player'] == player]
    df['X'] = (df['X'] / 100) * 105 * 100
    df['Y'] = (df['Y'] / 100) * 68 * 100
    total_shots = df.shape[0]
    total_goals = df[df['result'] == 'Goal'].shape[0]
    total_xG = df['xG'].sum()
    xG_per_shot = total_xG / total_shots
    points_average_distance = df['X'].mean()
    actual_average_distance = 105 - (df['X'] * 1.2).mean()

    if situation != None:
        ## OpenPlay
        df_openplay = df[df['situation'] == 'OpenPlay']
        total_shots_op = df_openplay.shape[0]
        total_goals_op = df_openplay[df_openplay['result'] == 'Goal'].shape[0]
        total_xG_op = df_openplay['xG'].sum()
        xG_per_shot_op = total_xG_op / total_shots_op
    
        ## FromCorner
        df_fromcorner = df[df['situation'] == 'FromCorner']
        total_shots_c = df_fromcorner.shape[0]
        total_goals_c = df_fromcorner[df_fromcorner['result'] == 'Goal'].shape[0]
        total_xG_c = df_fromcorner['xG'].sum()
        xG_per_shot_c = total_xG_c / total_shots_c
    
        ## SetPiece
        df_setpiece = df[df['situation'] == 'SetPiece']
        total_shots_sp = df_setpiece.shape[0]
        total_goals_sp = df_setpiece[df_setpiece['result'] == 'Goal'].shape[0]
        total_xG_sp = df_setpiece['xG'].sum()
        xG_per_shot_sp = total_xG_sp / total_shots_sp
    
        ## DirectFreekick
        df_freekick = df[df['situation'] == 'DirectFreekick']
        total_shots_fk = df_freekick.shape[0]
        total_goals_fk = df_freekick[df_freekick['result'] == 'Goal'].shape[0]
        total_xG_fk = df_freekick['xG'].sum()
        xG_per_shot_fk = total_xG_fk / total_shots_fk
    
        ## Penalty
        df_penalty = df[df['situation'] == 'Penalty']
        total_shots_p = df_penalty.shape[0]
        total_goals_p = df_penalty[df_penalty['result'] == 'Goal'].shape[0]
        total_xG_p = df_penalty['xG'].sum()
        xG_per_shot_p = total_xG_p / total_shots_p

    if shotType != None:
        ## RightFoot
        df_rf = df[df['shotType'] == 'RightFoot']
        total_shots_rf = df_rf.shape[0]
        total_goals_rf = df_rf[df_rf['result'] == 'Goal'].shape[0]
        total_xG_rf = df_rf['xG'].sum()
        xG_per_shot_rf = total_xG_rf / total_shots_rf

        ## LeftFoot
        df_lf = df[df['shotType'] == 'LeftFoot']
        total_shots_lf = df_lf.shape[0]
        total_goals_lf = df_lf[df_lf['result'] == 'Goal'].shape[0]
        total_xG_lf = df_lf['xG'].sum()
        xG_per_shot_lf = total_xG_lf / total_shots_lf

        ## Head
        df_h = df[df['shotType'] == 'Head']
        total_shots_h = df_h.shape[0]
        total_goals_h = df_h[df_h['result'] == 'Goal'].shape[0]
        total_xG_h = df_h['xG'].sum()
        xG_per_shot_h = total_xG_h / total_shots_h

        ## OtherBodyPart
        df_o = df[df['shotType'] == 'OtherBodyPart']
        total_shots_o = df_o.shape[0]
        total_goals_o = df_o[df_o['result'] == 'Goal'].shape[0]
        total_xG_o = df_o['xG'].sum()
        xG_per_shot_o = total_xG_o / total_shots_o

    
    pitch = VerticalPitch(
    pitch_type='uefa', 
    half=True, 
    pitch_color=background, 
    pad_bottom=.5, 
    line_color='white',
    linewidth=.5,
    axis=True, 
    label=True
    )

    # create a subplot with 2 rows and 1 column
    fig = plt.figure(figsize=(10, 12))
    fig.patch.set_facecolor(background)


    # Top row for the team names and score
    # [left, bottom, width, height]

    ax1 = fig.add_axes([0, 0.7, 1, .2])
    ax1.set_facecolor(background)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.text(
        x=0.5, 
        y=.85, 
        s=player, 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color=teamcolor, 
        ha='center'
    )
    ax1.text(
        x=0.5, 
        y=.7, 
        s=f'All shots in the {league} {season}', 
        fontsize=14,
        fontweight='bold',
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )
    ax1.text(
        x=0.25, 
        y=0.5, 
        s=f'Low Quality Chance', 
        fontsize=12, 
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )

    # add a scatter point between the two texts
    ax1.scatter(
        x=0.37, 
        y=0.53, 
        s=100, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.42, 
        y=0.53, 
        s=200, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.48, 
        y=0.53, 
        s=300, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.54, 
        y=0.53, 
        s=400, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.6, 
        y=0.53, 
        s=500, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )

    ax1.text(
        x=0.75, 
        y=0.5, 
        s=f'High Quality Chance', 
        fontsize=12, 
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )


    ax1.text(
        x=0.45, 
        y=0.27, 
        s=f'Goal', 
        fontsize=10, 
        fontproperties=font_prop, 
        color='white', 
        ha='right'
    )
    ax1.scatter(
        x=0.47, 
        y=0.3, 
        s=100, 
        color=teamcolor, 
        edgecolor='white', 
        linewidth=.8,
        alpha=.7
    )


    ax1.scatter(
        x=0.53, 
        y=0.3, 
        s=100, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )

    ax1.text(
        x=0.55, 
        y=0.27, 
        s=f'No Goal', 
        fontsize=10, 
        fontproperties=font_prop, 
        color='white', 
        ha='left'
    )

    ax1.set_axis_off()

    
    ax2 = fig.add_axes([.05, 0.25, .9, .5])
    ax2.set_facecolor(background)
    
    pitch.draw(ax=ax2)

    if situation != None:
        if situation == "all":
            for x in df.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif situation == "OpenPlay":
            for x in df_openplay.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif situation == "FromCorner":
            for x in df_fromcorner.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif situation == "SetPiece":
            for x in df_setpiece.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
    
        elif situation == "DirectFreekick":
            for x in df_freekick.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif situation == "Penalty":
            for x in df_penalty.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )

    if shotType != None:
        if shotType == "RightFoot":
            for x in df_rf.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif shotType == "LeftFoot":
            for x in df_lf.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif shotType == "Head":
            for x in df_h.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        elif shotType == "Other":
            for x in df_o.to_dict(orient='records'):
                pitch.scatter(
                    x['X'], 
                    x['Y'], 
                    s=400 * x['xG'], 
                    color=teamcolor if x['result'] == 'Goal' else background, 
                    ax=ax2,
                    alpha=.7,
                    linewidth=.8,
                    edgecolor='white'
                )
        
    ax2.set_axis_off()
    
    # add another axis for the stats
    ax3 = fig.add_axes([0, .2, 1, .05])
    ax3.set_facecolor(background)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.text(
        x=0.25, 
        y=.5, 
        s='Shots', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )
    shots_text = ''
    goals_text = ''
    xG_text = 0
    xG_per_shot_text = 0
    if situation != None:
        if situation == 'OpenPlay':
            shots_text = total_shots_op
        elif situation == 'FromCorner':
            shots_text = total_shots_c
        elif situation == 'SetPiece':
            shots_text = total_shots_sp
        elif situation == 'DirectFreekick':
            shots_text = total_shots_fk
        elif situation == 'Penalty':
            shots_text = total_shots_p
        else:
            shots_text = total_shots

    if shotType != None:
        if shotType == 'RightFoot':
            shots_text = total_shots_rf
        elif shotType == 'LeftFoot':
            shots_text = total_shots_lf
        elif shotType == 'Head':
            shots_text = total_shots_h
        elif shotType == 'Others':
            shots_text = total_shots_o
        else:
            shots_text = total_shots
    
    ax3.text(
        x=0.25, 
        y=0, 
        s=f'{shots_text}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.38, 
        y=.5, 
        s='Goals', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation != None:
        if situation == 'OpenPlay':
            goals_text = total_goals_op
        elif situation == 'FromCorner':
            goals_text = total_goals_c
        elif situation == 'SetPiece':
            goals_text = total_goals_sp
        elif situation == 'DirectFreekick':
            goals_text = total_goals_fk
        elif situation == 'Penalty':
            goals_text = total_goals_p
        else:
            goals_text = total_goals

    if shotType != None:
        if shotType == 'RightFoot':
            goals_text = total_goals_rf
        elif shotType == 'LeftFoot':
            goals_text = total_goals_lf
        elif shotType == 'Head':
            goals_text = total_goals_h
        elif shotType == 'Others':
            goals_text = total_goals_o
        else:
            goals_text = total_goals
    
    ax3.text(
        x=0.38, 
        y=0, 
        s=f'{goals_text}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.53, 
        y=.5, 
        s='xG', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation != None:
        if situation == 'OpenPlay':
            xG_text = total_xG_op
        elif situation == 'FromCorner':
            xG_text = total_xG_c
        elif situation == 'SetPiece':
            xG_text = total_xG_sp
        elif situation == 'DirectFreekick':
            xG_text = total_xG_fk
        elif situation == 'Penalty':
            xG_text = total_xG_p
        else:
            xG_text = total_xG

    if shotType != None:
        if shotType == 'RightFoot':
            xG_text = total_xG_rf
        elif shotType == 'LeftFoot':
            xG_text = total_xG_lf
        elif shotType == 'Head':
            xG_text = total_xG_h
        elif shotType == 'Others':
            xG_text = total_xG_o
        else:
            xG_text = total_xG
    ax3.text(
        x=0.53, 
        y=0, 
        s=f'{xG_text:.2f}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.63, 
        y=.5, 
        s='xG/Shot', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation != None:
        if situation == 'OpenPlay':
            xG_per_shot_text = xG_per_shot_op
        elif situation == 'FromCorner':
            xG_per_shot_text = xG_per_shot_c
        elif situation == 'SetPiece':
            xG_per_shot_text = xG_per_shot_sp
        elif situation == 'DirectFreekick':
            xG_per_shot_text = xG_per_shot_fk
        elif situation == 'Penalty':
            xG_per_shot_text = xG_per_shot_p
        else:
            xG_per_shot_text = xG_per_shot

    if shotType != None:
        if shotType == 'RightFoot':
            xG_per_shot_text = xG_per_shot_rf
        elif shotType == 'LeftFoot':
            xG_per_shot_text = xG_per_shot_lf
        elif shotType == 'Head':
            xG_per_shot_text = xG_per_shot_h
        elif shotType == 'Others':
            xG_per_shot_text = xG_per_shot_o
        else:
            xG_per_shot = xG_per_shot
    
    ax3.text(
        x=0.63, 
        y=0, 
        s=f'{xG_per_shot_text:.2f}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.set_axis_off()

    st.pyplot(fig)

def plot_shotmap_understat_conceded(df,team,league,teamcolor,situation,season):
    df11 = df[((df['h_team'] == team) & (df['h_a'] == 'a')) & (df['result'] != 'OwnGoal')]
    df21 = df[((df['a_team'] == team) & (df['h_a'] == 'h')) & (df['result'] != 'OwnGoal')]
    df1 = pd.concat([df11, df21], ignore_index=True)

    df12 = df[((df['h_team'] == team) & (df['h_a'] == 'h')) & (df['result'] == 'OwnGoal')]
    df22 = df[((df['a_team'] == team) & (df['h_a'] == 'a')) & (df['result'] == 'OwnGoal')]
    df2 = pd.concat([df12, df22], ignore_index=True)

    df = pd.concat([df1, df2], ignore_index=True)
    df['X'] = (df['X'] / 100) * 105 * 100
    df['Y'] = (df['Y'] / 100) * 68 * 100
    total_shots = df.shape[0]
    total_goals = df[(df['result'] == 'Goal') | (df['result'] == 'OwnGoal')].shape[0]
    total_xG = df['xG'].sum()
    xG_per_shot = total_xG / total_shots
    points_average_distance = df['X'].mean()
    actual_average_distance = 105 - (df['X'] * 1.2).mean()

    ## OpenPlay
    df_openplay = df[df['situation'] == 'OpenPlay']
    total_shots_op = df_openplay.shape[0]
    total_goals_op = df_openplay[df_openplay['result'] == 'Goal'].shape[0]
    total_xG_op = df_openplay['xG'].sum()
    xG_per_shot_op = total_xG_op / total_shots_op

    ## FromCorner
    df_fromcorner = df[df['situation'] == 'FromCorner']
    total_shots_c = df_fromcorner.shape[0]
    total_goals_c = df_fromcorner[df_fromcorner['result'] == 'Goal'].shape[0]
    total_xG_c = df_fromcorner['xG'].sum()
    xG_per_shot_c = total_xG_c / total_shots_c

    ## SetPiece
    df_setpiece = df[df['situation'] == 'SetPiece']
    total_shots_sp = df_setpiece.shape[0]
    total_goals_sp = df_setpiece[df_setpiece['result'] == 'Goal'].shape[0]
    total_xG_sp = df_setpiece['xG'].sum()
    xG_per_shot_sp = total_xG_sp / total_shots_sp

    ## DirectFreekick
    df_freekick = df[df['situation'] == 'DirectFreekick']
    total_shots_fk = df_freekick.shape[0]
    total_goals_fk = df_freekick[df_freekick['result'] == 'Goal'].shape[0]
    total_xG_fk = df_freekick['xG'].sum()
    xG_per_shot_fk = total_xG_fk / total_shots_fk

    ## Penalty
    df_penalty = df[df['situation'] == 'Penalty']
    total_shots_p = df_penalty.shape[0]
    total_goals_p = df_penalty[df_penalty['result'] == 'Goal'].shape[0]
    total_xG_p = df_penalty['xG'].sum()
    xG_per_shot_p = total_xG_p / total_shots_p
    
    pitch = VerticalPitch(
    pitch_type='uefa', 
    half=True, 
    pitch_color=background, 
    pad_bottom=.5, 
    line_color='white',
    linewidth=.5,
    axis=True, 
    label=True
    )

    # create a subplot with 2 rows and 1 column
    fig = plt.figure(figsize=(10, 12))
    fig.patch.set_facecolor(background)


    # Top row for the team names and score
    # [left, bottom, width, height]

    ax1 = fig.add_axes([0, 0.7, 1, .2])
    ax1.set_facecolor(background)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.text(
        x=0.5, 
        y=.85, 
        s=team, 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color=teamcolor, 
        ha='center'
    )
    ax1.text(
        x=0.5, 
        y=.7, 
        s=f'All shots conceded in the {league} {season}', 
        fontsize=14,
        fontweight='bold',
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )
    ax1.text(
        x=0.25, 
        y=0.5, 
        s=f'Low Quality Chance', 
        fontsize=12, 
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )

    # add a scatter point between the two texts
    ax1.scatter(
        x=0.37, 
        y=0.53, 
        s=100, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.42, 
        y=0.53, 
        s=200, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.48, 
        y=0.53, 
        s=300, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.54, 
        y=0.53, 
        s=400, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.6, 
        y=0.53, 
        s=500, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )

    ax1.text(
        x=0.75, 
        y=0.5, 
        s=f'High Quality Chance', 
        fontsize=12, 
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )


    ax1.text(
        x=0.45, 
        y=0.27, 
        s=f'Goal', 
        fontsize=10, 
        fontproperties=font_prop, 
        color='white', 
        ha='right'
    )
    ax1.scatter(
        x=0.47, 
        y=0.3, 
        s=100, 
        color=teamcolor, 
        edgecolor='white', 
        linewidth=.8,
        alpha=.7
    )


    ax1.scatter(
        x=0.53, 
        y=0.3, 
        s=100, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )

    ax1.text(
        x=0.55, 
        y=0.27, 
        s=f'No Goal', 
        fontsize=10, 
        fontproperties=font_prop, 
        color='white', 
        ha='left'
    )

    ax1.set_axis_off()

    
    ax2 = fig.add_axes([.05, 0.25, .9, .5])
    ax2.set_facecolor(background)
    
    pitch.draw(ax=ax2)


    if situation == "all":
        for x in df.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "OpenPlay":
        for x in df_openplay.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "FromCorner":
        for x in df_fromcorner.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "SetPiece":
        for x in df_setpiece.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )

    elif situation == "DirectFreekick":
        for x in df_freekick.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "Penalty":
        for x in df_penalty.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
        
    ax2.set_axis_off()
    
    # add another axis for the stats
    ax3 = fig.add_axes([0, .2, 1, .05])
    ax3.set_facecolor(background)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.text(
        x=0.25, 
        y=.5, 
        s='Shots', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        shots_text = total_shots_op
    elif situation == 'FromCorner':
        shots_text = total_shots_c
    elif situation == 'SetPiece':
        shots_text = total_shots_sp
    elif situation == 'DirectFreekick':
        shots_text = total_shots_fk
    elif situation == 'Penalty':
        shots_text = total_shots_p
    else:
        shots_text = total_shots
    
    ax3.text(
        x=0.25, 
        y=0, 
        s=f'{shots_text}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.38, 
        y=.5, 
        s='Goals', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        goals_text = total_goals_op
    elif situation == 'FromCorner':
        goals_text = total_goals_c
    elif situation == 'SetPiece':
        goals_text = total_goals_sp
    elif situation == 'DirectFreekick':
        goals_text = total_goals_fk
    elif situation == 'Penalty':
        goals_text = total_goals_p
    else:
        goals_text = total_goals
    ax3.text(
        x=0.38, 
        y=0, 
        s=f'{goals_text}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.53, 
        y=.5, 
        s='xG', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        xG_text = total_xG_op
    elif situation == 'FromCorner':
        xG_text = total_xG_c
    elif situation == 'SetPiece':
        xG_text = total_xG_sp
    elif situation == 'DirectFreekick':
        xG_text = total_xG_fk
    elif situation == 'Penalty':
        xG_text = total_xG_p
    else:
        xG_text = total_xG
    ax3.text(
        x=0.53, 
        y=0, 
        s=f'{xG_text:.2f}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.63, 
        y=.5, 
        s='xG/Shot', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        xG_per_shot_text = xG_per_shot_op
    elif situation == 'FromCorner':
        xG_per_shot_text = xG_per_shot_c
    elif situation == 'SetPiece':
        xG_per_shot_text = xG_per_shot_sp
    elif situation == 'DirectFreekick':
        xG_per_shot_text = xG_per_shot_fk
    elif situation == 'Penalty':
        xG_per_shot_text = xG_per_shot_p
    else:
        xG_per_shot_text = xG_per_shot
    
    ax3.text(
        x=0.63, 
        y=0, 
        s=f'{xG_per_shot_text:.2f}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.set_axis_off()
    st.pyplot(fig)

def plot_shotmap_understat_team(df_shots,team,league,teamcolor,situation,season):
    df1 = df_shots[(df_shots['h_team'] == team) & (df_shots['h_a'] == 'h') & (df_shots['result'] != 'OwnGoal')]
    df2 = df_shots[(df_shots['a_team'] == team) & (df_shots['h_a'] == 'a') & (df_shots['result'] != 'OwnGoal')]
    df = pd.concat([df1, df2], ignore_index=True)
    df['X'] = (df['X'] / 100) * 105 * 100
    df['Y'] = (df['Y'] / 100) * 68 * 100
    total_shots = df.shape[0]
    total_goals = df[df['result'] == 'Goal'].shape[0]
    total_xG = df['xG'].sum()
    xG_per_shot = total_xG / total_shots
    points_average_distance = df['X'].mean()
    actual_average_distance = 105 - (df['X'] * 1.2).mean()

    top_5 = df.groupby('player')['id'].count()

    # Sort in descending order and get the top 5
    top_5_shooters = top_5.sort_values(ascending=False).head(5)
    top_5_shooters_df = top_5_shooters.reset_index()
    top_5_shooters_df.columns = ['player', 'number_of_shots']

    ## OpenPlay
    df_openplay = df[df['situation'] == 'OpenPlay']
    total_shots_op = df_openplay.shape[0]
    total_goals_op = df_openplay[df_openplay['result'] == 'Goal'].shape[0]
    total_xG_op = df_openplay['xG'].sum()
    xG_per_shot_op = total_xG_op / total_shots_op

    top_5_op = df_openplay.groupby('player')['id'].count()
    top_5_shooters_op = top_5_op.sort_values(ascending=False).head(5)
    top_5_shooters_df_op = top_5_shooters_op.reset_index()
    top_5_shooters_df_op.columns = ['player', 'number_of_shots']

    ## FromCorner
    df_fromcorner = df[df['situation'] == 'FromCorner']
    total_shots_c = df_fromcorner.shape[0]
    total_goals_c = df_fromcorner[df_fromcorner['result'] == 'Goal'].shape[0]
    total_xG_c = df_fromcorner['xG'].sum()
    xG_per_shot_c = total_xG_c / total_shots_c

    top_5_fc = df_fromcorner.groupby('player')['id'].count()
    top_5_shooters_fc = top_5_fc.sort_values(ascending=False).head(5)
    top_5_shooters_df_fc = top_5_shooters_fc.reset_index()
    top_5_shooters_df_fc.columns = ['player', 'number_of_shots']

    ## SetPiece
    df_setpiece = df[df['situation'] == 'SetPiece']
    total_shots_sp = df_setpiece.shape[0]
    total_goals_sp = df_setpiece[df_setpiece['result'] == 'Goal'].shape[0]
    total_xG_sp = df_setpiece['xG'].sum()
    xG_per_shot_sp = total_xG_sp / total_shots_sp

    top_5_sp = df_setpiece.groupby('player')['id'].count()
    top_5_shooters_sp = top_5_sp.sort_values(ascending=False).head(5)
    top_5_shooters_df_sp = top_5_shooters_sp.reset_index()
    top_5_shooters_df_sp.columns = ['player', 'number_of_shots']

    ## DirectFreekick
    df_freekick = df[df['situation'] == 'DirectFreekick']
    total_shots_fk = df_freekick.shape[0]
    total_goals_fk = df_freekick[df_freekick['result'] == 'Goal'].shape[0]
    total_xG_fk = df_freekick['xG'].sum()
    xG_per_shot_fk = total_xG_fk / total_shots_fk

    top_5_fk = df_freekick.groupby('player')['id'].count()
    top_5_shooters_fk = top_5_fk.sort_values(ascending=False).head(5)
    top_5_shooters_df_fk = top_5_shooters_fk.reset_index()
    top_5_shooters_df_fk.columns = ['player', 'number_of_shots']

    ## Penalty
    df_penalty = df[df['situation'] == 'Penalty']
    total_shots_p = df_penalty.shape[0]
    total_goals_p = df_penalty[df_penalty['result'] == 'Goal'].shape[0]
    total_xG_p = df_penalty['xG'].sum()
    xG_per_shot_p = total_xG_p / total_shots_p

    top_5_p = df_penalty.groupby('player')['id'].count()
    top_5_shooters_p = top_5_p.sort_values(ascending=False).head(5)
    top_5_shooters_df_p = top_5_shooters_p.reset_index()
    top_5_shooters_df_p.columns = ['player', 'number_of_shots']
    
    pitch = VerticalPitch(
    pitch_type='uefa', 
    half=True, 
    pitch_color=background, 
    pad_bottom=.5, 
    line_color='white',
    linewidth=.5,
    axis=True, 
    label=True
    )

    # create a subplot with 2 rows and 1 column
    fig = plt.figure(figsize=(9, 10))
    fig.patch.set_facecolor(background)

    # Top row for the team names and score
    # [left, bottom, width, height]

    ax1 = fig.add_axes([0, 0.7, 1, .2])
    ax1.set_facecolor(background)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.text(
        x=0.5, 
        y=.85, 
        s=team, 
        fontsize=25, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color=teamcolor, 
        ha='center'
    )
    ax1.text(
        x=0.5, 
        y=.7, 
        s=f'All shots in the {league} {season}', 
        fontsize=16,
        fontweight='bold',
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )
    ax1.text(
        x=0.25, 
        y=0.5, 
        s=f'Low Quality Chance', 
        fontsize=12, 
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )

    # add a scatter point between the two texts
    ax1.scatter(
        x=0.37, 
        y=0.53, 
        s=100, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.42, 
        y=0.53, 
        s=200, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.48, 
        y=0.53, 
        s=300, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.54, 
        y=0.53, 
        s=400, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )
    ax1.scatter(
        x=0.6, 
        y=0.53, 
        s=500, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )

    ax1.text(
        x=0.75, 
        y=0.5, 
        s=f'High Quality Chance', 
        fontsize=12, 
        fontproperties=font_prop, 
        color='white', 
        ha='center'
    )


    ax1.text(
        x=0.45, 
        y=0.27, 
        s=f'Goal', 
        fontsize=15, 
        fontproperties=font_prop, 
        color='white', 
        ha='right'
    )
    ax1.scatter(
        x=0.47, 
        y=0.3, 
        s=100, 
        color=teamcolor, 
        edgecolor='white', 
        linewidth=.8,
        alpha=.7
    )


    ax1.scatter(
        x=0.53, 
        y=0.3, 
        s=100, 
        color=background, 
        edgecolor='white', 
        linewidth=.8
    )

    ax1.text(
        x=0.55, 
        y=0.27, 
        s=f'No Goal', 
        fontsize=15, 
        fontproperties=font_prop, 
        color='white', 
        ha='left'
    )

    ax1.set_axis_off()

    
    ax2 = fig.add_axes([.05, 0.25, .9, .5])
    ax2.set_facecolor(background)
    
    pitch.draw(ax=ax2)

    if situation == "all":
        for x in df.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "OpenPlay":
        for x in df_openplay.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "FromCorner":
        for x in df_fromcorner.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "SetPiece":
        for x in df_setpiece.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )

    elif situation == "DirectFreekick":
        for x in df_freekick.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
    elif situation == "Penalty":
        for x in df_penalty.to_dict(orient='records'):
            pitch.scatter(
                x['X'], 
                x['Y'], 
                s=400 * x['xG'], 
                color=teamcolor if x['result'] == 'Goal' else background, 
                ax=ax2,
                alpha=.7,
                linewidth=.8,
                edgecolor='white'
            )
        
    ax2.set_axis_off()
    
    # add another axis for the stats
    ax3 = fig.add_axes([0, .2, 1, .05])
    ax3.set_facecolor(background)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    ax3.text(
        x=0.25, 
        y=.5, 
        s='Shots', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        shots_text = total_shots_op
    elif situation == 'FromCorner':
        shots_text = total_shots_c
    elif situation == 'SetPiece':
        shots_text = total_shots_sp
    elif situation == 'DirectFreekick':
        shots_text = total_shots_fk
    elif situation == 'Penalty':
        shots_text = total_shots_p
    else:
        shots_text = total_shots
    
    ax3.text(
        x=0.25, 
        y=0, 
        s=f'{shots_text}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.38, 
        y=.5, 
        s='Goals', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        goals_text = total_goals_op
    elif situation == 'FromCorner':
        goals_text = total_goals_c
    elif situation == 'SetPiece':
        goals_text = total_goals_sp
    elif situation == 'DirectFreekick':
        goals_text = total_goals_fk
    elif situation == 'Penalty':
        goals_text = total_goals_p
    else:
        goals_text = total_goals
    ax3.text(
        x=0.38, 
        y=0, 
        s=f'{goals_text}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.53, 
        y=.5, 
        s='xG', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        xG_text = total_xG_op
    elif situation == 'FromCorner':
        xG_text = total_xG_c
    elif situation == 'SetPiece':
        xG_text = total_xG_sp
    elif situation == 'DirectFreekick':
        xG_text = total_xG_fk
    elif situation == 'Penalty':
        xG_text = total_xG_p
    else:
        xG_text = total_xG
    ax3.text(
        x=0.53, 
        y=0, 
        s=f'{xG_text:.2f}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.text(
        x=0.63, 
        y=.5, 
        s='xG/Shot', 
        fontsize=20, 
        fontproperties=font_prop, 
        fontweight='bold', 
        color='white', 
        ha='left'
    )

    if situation == 'OpenPlay':
        xG_per_shot_text = xG_per_shot_op
    elif situation == 'FromCorner':
        xG_per_shot_text = xG_per_shot_c
    elif situation == 'SetPiece':
        xG_per_shot_text = xG_per_shot_sp
    elif situation == 'DirectFreekick':
        xG_per_shot_text = xG_per_shot_fk
    elif situation == 'Penalty':
        xG_per_shot_text = xG_per_shot_p
    else:
        xG_per_shot_text = xG_per_shot
    
    ax3.text(
        x=0.63, 
        y=0, 
        s=f'{xG_per_shot_text:.2f}', 
        fontsize=16, 
        fontproperties=font_prop, 
        color=teamcolor, 
        ha='left'
    )

    ax3.set_axis_off()

    ax4 = fig.add_axes([0.85, 0.1, 0.15, 0.5])  # Adjust position and size as needed
    ax4.set_facecolor(background)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    if situation == 'OpenPlay':
        for i, (player, shots) in enumerate(zip(top_5_shooters_df_op['player'], top_5_shooters_df_op['number_of_shots'])):
            y_pos = 1 - (i * 0.12)  # Adjust spacing between players
            ax4.text(0.5, y_pos, player.split()[-1], fontsize=15, fontproperties=font_prop, color='white', ha='center')
            ax4.text(0.5, y_pos - 0.05, f'{shots} shots', fontsize=12, fontproperties=font_prop, color=teamcolor, ha='center')
    elif situation == 'FromCorner':
        for i, (player, shots) in enumerate(zip(top_5_shooters_df_fc['player'], top_5_shooters_df_fc['number_of_shots'])):
            y_pos = 1 - (i * 0.12)  # Adjust spacing between players
            ax4.text(0.5, y_pos, player.split()[-1], fontsize=15, fontproperties=font_prop, color='white', ha='center')
            ax4.text(0.5, y_pos - 0.05, f'{shots} shots', fontsize=12, fontproperties=font_prop, color=teamcolor, ha='center')
    elif situation == 'SetPiece':
        for i, (player, shots) in enumerate(zip(top_5_shooters_df_sp['player'], top_5_shooters_df_sp['number_of_shots'])):
            y_pos = 1 - (i * 0.12)  # Adjust spacing between players
            ax4.text(0.5, y_pos, player.split()[-1], fontsize=15, fontproperties=font_prop, color='white', ha='center')
            ax4.text(0.5, y_pos - 0.05, f'{shots} shots', fontsize=12, fontproperties=font_prop, color=teamcolor, ha='center')
    elif situation == 'DirectFreekick':
        for i, (player, shots) in enumerate(zip(top_5_shooters_df_fk['player'], top_5_shooters_df_fk['number_of_shots'])):
            y_pos = 1 - (i * 0.12)  # Adjust spacing between players
            ax4.text(0.5, y_pos, player.split()[-1], fontsize=15, fontproperties=font_prop, color='white', ha='center')
            ax4.text(0.5, y_pos - 0.05, f'{shots} shots', fontsize=12, fontproperties=font_prop, color=teamcolor, ha='center')
    elif situation == 'Penalty':
        for i, (player, shots) in enumerate(zip(top_5_shooters_df_p['player'], top_5_shooters_df_p['number_of_shots'])):
            y_pos = 1 - (i * 0.12)  # Adjust spacing between players
            ax4.text(0.5, y_pos, player.split()[-1], fontsize=15, fontproperties=font_prop, color='white', ha='center')
            ax4.text(0.5, y_pos - 0.05, f'{shots} shots', fontsize=12, fontproperties=font_prop, color=teamcolor, ha='center')
    else:
        for i, (player, shots) in enumerate(zip(top_5_shooters_df['player'], top_5_shooters_df['number_of_shots'])):
            y_pos = 1 - (i * 0.12)  # Adjust spacing between players
            ax4.text(0.5, y_pos, player.split()[-1], fontsize=15, fontproperties=font_prop, color='white', ha='center')
            ax4.text(0.5, y_pos - 0.05, f'{shots} shots', fontsize=12, fontproperties=font_prop, color=teamcolor, ha='center')
    
    # Plot each player's name and their shot count
    
    
    ax4.set_axis_off()

    st.pyplot(fig)

def summarize_player_shots(df):
    # Filter relevant shot types
    mask_shots = df['type'].isin(['Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'])
    shots_df = df[mask_shots].copy()

    # Create new boolean columns
    shots_df['BlockedShots'] = ((shots_df['type'] == 'SavedShot') & (shots_df['shotBlocked'] == True)).astype(int)
    shots_df['CleanSavedShot'] = ((shots_df['type'] == 'SavedShot') & (shots_df['shotBlocked'] == False)).astype(int)

    # Summarize main shot types
    player_summary = shots_df.groupby('playerName')['type'].value_counts().unstack(fill_value=0)
    player_summary = player_summary.rename(columns={
        'Goal': 'Goals',
        'MissedShots': 'Off Target',
        'ShotOnPost': 'Woodwork',
    })

    # Add 'On Target' from CleanSavedShot column
    saved_shots = shots_df.groupby('playerName')['CleanSavedShot'].sum().rename('On Target')
    player_summary = player_summary.merge(saved_shots, left_index=True, right_index=True, how='left')

    # Add BlockedShots
    blocked = shots_df.groupby('playerName')['BlockedShots'].sum().rename('BlockedShots')
    player_summary = player_summary.merge(blocked, left_index=True, right_index=True, how='left')

    # Fill missing columns
    for col in ['Goals', 'On Target', 'Off Target', 'Woodwork', 'BlockedShots']:
        if col not in player_summary.columns:
            player_summary[col] = 0

    # Add total shots
    player_summary['Total Shots'] = (
        player_summary['Goals'] +
        player_summary['On Target'] +
        player_summary['Off Target'] +
        player_summary['Woodwork'] +
        player_summary['BlockedShots']
    )

    # Add total xG
    xg_summary = shots_df.groupby('playerName')['xG'].sum().rename('Total xG')

    # Round conditionally
    xg_summary = xg_summary.apply(lambda x: round(x, 2) if x < 0.9 else round(x, 3))

    # Merge and compute xG/Shot
    player_summary = player_summary.merge(xg_summary, left_index=True, right_index=True)
    player_summary['xG/Shot'] = (player_summary['Total xG'] / player_summary['Total Shots']).round(3)

    # Reset index
    player_summary = player_summary.reset_index()

    # Filter out players with no shots
    player_summary = player_summary[player_summary['Total Shots'] > 0]

    # Sort by total shots
    player_summary = player_summary.sort_values(by='Total Shots', ascending=False)

    return player_summary

def calculate_angle(x, y,GOAL_X,GOAL_Y):
    goal_width = 7.32
    goal_left = GOAL_Y - goal_width / 2
    goal_right = GOAL_Y + goal_width / 2
    a = np.sqrt((GOAL_X - x)**2 + (goal_left - y)**2)
    b = np.sqrt((GOAL_X - x)**2 + (goal_right - y)**2)
    c = goal_width
    try:
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    except:
        angle = 0
    return angle

def shotMap_ws(df,axs,fig,pitch,hteam,ateam,team1_facecolor,team2_facecolor,text_color,background,situation):

    xgb_model = joblib.load('C://Users//acer//Documents//GitHub//IndianCitizen//ScorePredict//notebooks//xgboost_xg_model.pkl')

    # Filter shot events
    shot_events = ['SavedShot', 'MissedShots', 'ShotOnPost', 'Goal']
    df = df[df['type'].isin(shot_events)].copy()

    # Compute distance
    GOAL_X = 105
    GOAL_Y = 34
    df['shot_distance'] = np.sqrt((GOAL_X - df['x'])**2 + (GOAL_Y - df['y'])**2)

    df['shot_angle'] = df.apply(lambda row: calculate_angle(row['x'], row['y'],GOAL_X,GOAL_Y), axis=1)

    # Fill missing shotBodyType and one-hot encode
    df['shotBodyType'] = df['shotBodyType'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['shotBodyType'], drop_first=True)

    
    # Use same features as training
    training_features = ['shot_distance', 'shot_angle', 'shotOpenPlay', 'shotCounter', 
                        'shotSetPiece', 'shotDirectCorner'] + \
                        [col for col in xgb_model.get_booster().feature_names if col.startswith('shotBodyType_')]

    # Add any missing columns (if some shotBodyType dummies didn’t appear in this match)
    for col in training_features:
        if col not in df.columns:
            df[col] = 0


    # Reorder columns
    X_match = df[training_features]

    # Ensure these columns are proper booleans
    bool_cols = ['shotOpenPlay', 'shotCounter', 'shotSetPiece', 'shotDirectCorner']
    for col in bool_cols:
        X_match[col] = X_match[col].astype(bool)

    df['xG'] = xgb_model.predict_proba(X_match)[:, 1]

    if situation == 'All':
        mask1 = ((df['teamName'] == hteam)) & ((df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost'))
        mask2 = ((df['teamName'] == ateam)) & ((df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost'))
    else:
        mask1 = ((df['teamName'] == hteam)) & (df['situation'] == situation) & ((df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost'))
        mask2 = ((df['teamName'] == ateam)) & (df['situation'] == situation) & ((df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost'))


    home_shots_df = df[mask1]
    home_shots_df.reset_index(drop=True, inplace=True)
    h_missed = home_shots_df[home_shots_df['type'] == 'MissedShots']
    h_saved = home_shots_df[(home_shots_df['type'] == 'SavedShot') & (home_shots_df['shotBlocked'] == False)]
    h_blocked = home_shots_df[(home_shots_df['type'] == 'SavedShot') & (home_shots_df['shotBlocked'] == True)]
    h_post = home_shots_df[home_shots_df['type'] == 'ShotOnPost']
    h_goals = home_shots_df[(home_shots_df['type'] == 'Goal') & (home_shots_df['goalOwn'] == False)]
    h_own_goals = home_shots_df[(home_shots_df['type'] == 'Goal') & (home_shots_df['goalOwn'] == True)]

    away_shots_df = df[mask2]
    away_shots_df.reset_index(drop=True, inplace=True)
    a_missed = away_shots_df[away_shots_df['type'] == 'MissedShots']
    a_saved = away_shots_df[(away_shots_df['type'] == 'SavedShot') & (away_shots_df['shotBlocked'] == False)]
    a_blocked = away_shots_df[(away_shots_df['type'] == 'SavedShot') & (away_shots_df['shotBlocked'] == True)]
    a_post = away_shots_df[away_shots_df['type'] == 'ShotOnPost']
    a_goals = away_shots_df[(away_shots_df['type'] == 'Goal') & (away_shots_df['goalOwn'] == False)]
    a_own_goals = away_shots_df[(away_shots_df['type'] == 'Goal') & (away_shots_df['goalOwn'] == True)]

    away_shots_df['x'] = pitch.dim.right - away_shots_df.x
    away_shots_df['y'] = pitch.dim.top - away_shots_df.y
    team1_hist_y = sns.kdeplot(y=away_shots_df.y, ax=axs['left'], color=team2_facecolor, fill=True)
    #team1_hist_x = sns.kdeplot(x=away_shots_df.x, ax=axs['top'], color=team2_facecolor, fill=True)
    #team2_hist_x = sns.kdeplot(x=home_shots_df.x, ax=axs['top'], color=team1_facecolor, fill=True)
    team2_hist_y = sns.kdeplot(y=home_shots_df.y, ax=axs['right'], color=team1_facecolor, fill=True)

    a_missed['x'] = pitch.dim.right - a_missed.x
    a_saved['x'] = pitch.dim.right - a_saved.x
    a_post['x'] = pitch.dim.right - a_post.x
    a_goals['x'] = pitch.dim.right - a_goals.x
    a_own_goals['x'] = pitch.dim.right - a_own_goals.x
    a_blocked['x'] = pitch.dim.right - a_blocked.x

    a_missed['y'] = pitch.dim.top - a_missed.y
    a_saved['y'] = pitch.dim.top - a_saved.y
    a_post['y'] = pitch.dim.top - a_post.y
    a_goals['y'] = pitch.dim.top - a_goals.y
    a_own_goals['y'] = pitch.dim.top - a_own_goals.y
    a_blocked['y'] = pitch.dim.top - a_blocked.y

    pitch.scatter(h_missed.x,h_missed.y,marker='o', edgecolors=team1_facecolor, s=7000 * h_missed['xG'],linewidth=3, c=background,ax=axs['pitch'])
    pitch.scatter(h_saved.x,h_saved.y,marker='o', edgecolors='white', s=7000 * h_saved['xG'], c=team1_facecolor,linewidth=3,zorder=4,ax=axs['pitch'])
    pitch.scatter(h_post.x,h_post.y,marker='o', edgecolors='green', s=7000 * h_post['xG'],linewidth=3, c=team1_facecolor,zorder=5,ax=axs['pitch'])
    pitch.scatter(h_goals.x,h_goals.y,marker='football', edgecolors=text_color, s=9000 * h_goals['xG'],zorder=6,linewidths=3, c=team1_facecolor,ax=axs['pitch'])
    pitch.scatter(a_own_goals.x,a_own_goals.y,marker='football', edgecolors=team2_facecolor, s=1000,zorder=6, c=team1_facecolor,ax=axs['pitch'])
    pitch.scatter(h_blocked.x,h_blocked.y,marker='s', edgecolors=team1_facecolor, s=5000 * h_blocked['xG'],zorder=5,linewidth=3, c=background,ax=axs['pitch'])


    pitch.scatter(a_missed.x,a_missed.y,marker='o', edgecolors=team2_facecolor, s=7000 * a_missed['xG'],linewidth=3, c=background,ax=axs['pitch'])
    pitch.scatter(a_saved.x,a_saved.y,marker='o', edgecolors='white', s=7000 * a_saved['xG'], c=team2_facecolor,linewidth=3,zorder=4,ax=axs['pitch'])
    pitch.scatter(a_post.x,a_post.y,marker='o', edgecolors='green', s=7000 * a_post['xG'], c=team2_facecolor,linewidth=3,zorder=5,ax=axs['pitch'])
    pitch.scatter(a_goals.x,a_goals.y,marker='football', edgecolors=text_color, s=9000 * a_goals['xG'],zorder=6,linewidths=3, c=team2_facecolor,ax=axs['pitch'])
    pitch.scatter(h_own_goals.x,h_own_goals.y,marker='football', edgecolors=team1_facecolor, s=1000,zorder=6, c=team2_facecolor,ax=axs['pitch'])
    pitch.scatter(a_blocked.x,a_blocked.y,marker='s', edgecolors=team2_facecolor, s=5000 * a_blocked['xG'],zorder=5,linewidth=3, c=background,ax=axs['pitch'])



    pitch.scatter(4,-5,marker='football', edgecolors=text_color, s=1000, c=background,ax=axs['pitch'])
    pitch.annotate('Goal', xy=(10,-5), fontsize=30,color=text_color,fontproperties=font_prop,ax=axs['pitch'], ha='center', va='center')
    
    pitch.scatter(23,-5,marker='o', edgecolors=background, s=1000, c=text_color,ax=axs['pitch'])
    pitch.annotate('On Target', xy=(33,-5), fontsize=30,color=text_color,fontproperties=font_prop,ax=axs['pitch'], ha='center', va='center')

    pitch.scatter(48,-5,marker='o', edgecolors='green',linewidth=5, s=1000, c=background,ax=axs['pitch'])
    pitch.annotate('Woodwork', xy=(58,-5), fontsize=30,color=text_color,fontproperties=font_prop,ax=axs['pitch'], ha='center', va='center')

    pitch.scatter(70,-5,marker='o', edgecolors=text_color, s=1000, c=background,ax=axs['pitch'])
    pitch.annotate('Off Target', xy=(78,-5), fontsize=30,color=text_color,fontproperties=font_prop,ax=axs['pitch'], ha='center', va='center')

    pitch.scatter(90,-5,marker='s', edgecolors=text_color,linewidth=3, s=1000, c=background,ax=axs['pitch'])
    pitch.annotate('Blocked', xy=(98,-5), fontsize=30,color=text_color,fontproperties=font_prop,ax=axs['pitch'], ha='center', va='center')

    #pitch.annotate('ShotMap', xy=(1, 75),fontproperties=font_prop, fontsize=80,color=text_color,ax=axs['pitch'], ha='left', va='center')

    hteam_img = mpimg.imread(f'C:\\Users\\acer\\Documents\\GitHub\\IndianCitizen\\ScorePredict\\Images\\TeamLogos\\{hteam}.png')

    ax_image = add_image(
        hteam_img, fig, left=0.35, bottom=0.65, width=0.12, height=0.12,aspect='equal'
    )

    ateam_img = mpimg.imread(f'C:\\Users\\acer\\Documents\\GitHub\\IndianCitizen\\ScorePredict\\Images\\TeamLogos\\{ateam}.png')

    ax_image = add_image(
        ateam_img, fig, left=0.52, bottom=0.65, width=0.12, height=0.12,aspect='equal'
    )

    h_xg = round(home_shots_df['xG'].sum(),2)
    a_xg = round(away_shots_df['xG'].sum(),2)
    summary_data = {
        'Team': [hteam, ateam],
        'Goals': [len(h_goals) + len(a_own_goals), len(a_goals) + len(h_own_goals)],
        'On Target': [len(h_goals) + len(h_saved), len(a_goals) + len(a_saved)],
        'Off Target': [len(h_missed), len(a_missed)],
        'Woodwork': [len(h_post), len(a_post)],
        'Blocked': [len(h_blocked), len(a_blocked)],
        'Own Goals': [len(h_own_goals), len(a_own_goals)],
        'xG': [h_xg, a_xg]
    }

    summary_df = pd.DataFrame(summary_data)
    player_df = summarize_player_shots(df)
    #print(summary_df)
    #st.dataframe(summary_df)
    return summary_df,player_df,home_shots_df,away_shots_df

def xgFlow(ax,home_shots_df,away_shots_df,team1,team2,team1_facecolor,team2_facecolor,text_color,background):
    home_shots_df = home_shots_df.sort_values(by='eventId')
    away_shots_df = away_shots_df.sort_values(by='eventId')
    dfhome_xG = home_shots_df[['playerName','minute','xG','type','situation','teamName']]
    dfaway_xG = away_shots_df[['playerName','minute','xG','type','situation','teamName']]
    
    df_xG = pd.concat([dfhome_xG, dfaway_xG], ignore_index=True)
    
    #df_xG['cumulative_xG'] = df_xG.groupby('teamName')['xG'].sum()
    df_xG['cumulative_xG'] = df_xG.groupby('teamName')['xG'].cumsum()

    
    df_xG['minute'] = pd.to_numeric(df_xG['minute'], errors='coerce')
    
    df_xG['half'] = df_xG['minute'].apply(lambda x: 1 if x <= 45 else 2)

    ax.set_facecolor(background)
    
    for team in df_xG['teamName'].unique():
        
        team_df = df_xG[df_xG['teamName'] == team]
        
        # add a 0 xG row at the start of the match
        team_df = pd.concat([pd.DataFrame({'teamName': team, 'minute': 0, 'xG': 0, 'type': 'Goal', 'cumulative_xG': 0, 'half': 1},
                                          index=[0]), team_df])
        
        # Also add a row at the beginning of the second half to make the lines start where the first half ended
        team_df = pd.concat([team_df[team_df['half'] == 1], pd.DataFrame({'teamName': team, 'minute': 45, 'xG': 0, 'type': 'Goal',
                                                                          'cumulative_xG': team_df[team_df['half'] == 1]['cumulative_xG'].iloc[-1],
                                                                          'half': 2}, index=[0]), team_df[team_df['half'] == 2]])
    
        for half in team_df['half'].unique():
            half_df = team_df[team_df['half'] == half]
            ax.plot(
                half_df['minute'], 
                half_df['cumulative_xG'], 
                label=team, 
                drawstyle='steps-post',
                c=team2_facecolor if team == team2 else team1_facecolor,
                linewidth=5
            )   
            
    
    # We Can add a scatter plot to show the goals
    for team in df_xG['teamName'].unique():
        team_df = df_xG[(df_xG['teamName'] == team) & (df_xG['type'] == 'Goal')].to_dict(orient='records')
        for x in team_df:
            ax.scatter(
                x['minute'], 
                x['cumulative_xG'], 
                c=team2_facecolor if team == team2 else team1_facecolor,
                edgecolor=text_color,
                s=800,
                marker='*',
                # We want the goals to be on top of the lines
                zorder=5
            )
            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin

            # Define a vertical offset as a percentage of the y-range
            offset_y = 0.08 * y_range
            # add a label to the goals for the player who scored
            ax.text(
                x['minute']+1, 
                x['cumulative_xG'] + offset_y, 
                f"{x['playerName']}\nxG: {round(x['xG'],2)}", 
                ha='center', 
                va='center',
                c=text_color,
                fontproperties=font_prop,
                fontsize=22,
                zorder=10
            )
            
    # Let's label the x axis with first and second half
    ax.set_xticks([0, 45, 90])
    ax.set_xticklabels(['0\'', '45\'', '90\''])
    
    #ax.text(-0.9, df_xG['cumulative_xG'].max() + 0.12, 'xG Flow', ha='left',fontproperties=font_prop, fontsize=50,color='white')
    
    # Let's label the y axis with the cumulative xG
    ax.set_ylabel('Cumulative xG', fontfamily='monospace',fontproperties=font_prop, fontsize=22,color=text_color)
    
    # Let's get rid of the right and top spines
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(text_color)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color(text_color)

    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_color(text_color)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(text_color)
    
    ax.tick_params(axis='x', colors=text_color, labelsize=15)
    ax.tick_params(axis='y', colors=text_color, labelsize=15)
    
    ax.axvline(x=45, color=text_color, linestyle='--', linewidth=1,alpha=0.5)

def get_passes_df(df):
    df1 = df[~df['type'].str.contains('SubstitutionOn|FormationChange|FormationSet|Card')]
    df = df1
    df.loc[:, "receiver"] = df["playerId"].shift(-1)
    passes_ids = df.index[df['type'] == 'Pass']
    df_passes = df.loc[passes_ids, ["index", "x", "y","minute", "endX", "endY", "teamName", "playerId", "receiver", "type", "outcomeType","isFirstEleven","playerName"]].copy()

    return df_passes

def get_initials(name):
    if pd.isna(name):
        return ''
    parts = name.strip().split()
    initials = [p[0].upper() + '.' for p in parts if p]
    return ''.join(initials)

def get_pass_matrix(passes_df, teamName):
    team_passes = passes_df[passes_df["teamName"] == teamName].copy()

    # Clean and convert names to initials
    team_passes['playerName'] = team_passes['playerName'].apply(unidecode)
    team_passes['playerName'] = team_passes['playerName'].apply(get_initials)


    # Build mapping AFTER cleaning
    id_to_name = dict(zip(team_passes['playerId'], team_passes['playerName']))

    # Map receiverId to initials using cleaned playerName mapping
    team_passes['receiverName'] = team_passes['receiver'].map(id_to_name)

    # Drop any rows with missing receiverName (if any)
    team_passes = team_passes.dropna(subset=['receiverName'])

    # Create pass matrix
    pass_matrix = team_passes.groupby(['playerName', 'receiverName']).size().unstack(fill_value=0)

    return pass_matrix

def filter_passes_for_subwindow(match_df, passes_df, teamName, minute_start, minute_end):
    # Get starting players
    starting_players = match_df[
        (match_df['teamName'] == teamName) &
        (match_df['isFirstEleven'] == True)
    ]['playerId'].unique().tolist()

    # Players subbed on before the end of this window
    subs_on = match_df[
        (match_df['teamName'] == teamName) &
        (match_df['type'] == 'SubstitutionOn') &
        (match_df['minute'] < minute_end)
    ]['playerId'].tolist()

    # Players subbed off before the start of this window
    subs_off = match_df[
        (match_df['teamName'] == teamName) &
        (match_df['type'] == 'SubstitutionOff') &
        (match_df['minute'] < minute_start)
    ]['playerId'].tolist()

    # Calculate who is on the pitch in this time window
    valid_players = set(starting_players + subs_on) - set(subs_off)

    # Filter the passes
    passes_filtered = passes_df[
        (passes_df['teamName'] == teamName) &
        (passes_df['minute'] >= minute_start) &
        (passes_df['minute'] < minute_end) &
        (passes_df['playerId'].isin(valid_players)) &
        (passes_df['receiver'].isin(valid_players))
    ]

    return passes_filtered

def get_passes_between_df(teamName, passes_df):
    passes_df = passes_df[passes_df["teamName"] == teamName].copy()
    # Group by playerId and aggregate pass location + isFirstEleven
    average_locs_and_count_df = (
        passes_df.groupby('playerId').agg({
            'x': 'median',
            'y': 'median',
            'playerId': 'count',
            'playerName': 'first',
            'isFirstEleven': 'first'
        })
    )

    average_locs_and_count_df.columns = ['pass_avg_x', 'pass_avg_y', 'count', 'playerName', 'isFirstEleven']
    average_locs_and_count_df.index.name = 'playerId'

    average_locs_and_count_df['playerName'] = average_locs_and_count_df['playerName'].apply(unidecode)

    # Build pass combinations between players (order-independent)
    passes_player_ids_df = passes_df[['index', 'playerId', 'receiver', 'teamName']].copy()
    passes_player_ids_df['pos_max'] = passes_player_ids_df[['playerId', 'receiver']].max(axis=1)
    passes_player_ids_df['pos_min'] = passes_player_ids_df[['playerId', 'receiver']].min(axis=1)

    passes_between_df = (
        passes_player_ids_df
        .groupby(['pos_min', 'pos_max'])['index']
        .count()
        .reset_index()
        .rename(columns={'index': 'pass_count'})
    )

    # Merge in player locations (start and end)
    passes_between_df = passes_between_df.merge(
        average_locs_and_count_df, left_on='pos_min', right_index=True
    ).merge(
        average_locs_and_count_df, left_on='pos_max', right_index=True, suffixes=['', '_end']
    )

    return passes_between_df, average_locs_and_count_df

def calculate_centralization_index(teamName, passes_df):

    # Filter passes within the given time range and team
    passes_df = passes_df[(passes_df["teamName"] == teamName)]

    # Calculate number of passes made by each player
    player_passes_count = passes_df["playerId"].value_counts()

    # Get the total number of passes made by the team
    total_passes = player_passes_count.sum()

    # Get the maximal number of passes made by a single player
    max_passes = player_passes_count.max()

    # Number of players in the team
    num_players = len(player_passes_count)

    # Calculate the denominator (Total passes * (Number of players - 1))
    denominator = total_passes * 10

    # Calculate the numerator (sum of (Max passes - Player passes))
    numerator = sum(max_passes - player_passes_count)

    # Calculate the centralization index
    centralization_index = numerator / denominator if denominator != 0 else 0

    return centralization_index

def pass_network_visualization(ax,df, passes_between_df, average_locs_and_count_df,text_color,background, col, teamName,MAX_LINE_WIDTH,flipped,ci):
    MAX_MARKER_SIZE = 6000
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max()) * MAX_LINE_WIDTH
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']/ average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE) #You can plot variable size of each player's node according to their passing volume, in the plot using this
    MIN_TRANSPARENCY = 0.55
    MAX_TRANSPARENCY = 0.95
    color = np.array(to_rgba(col))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=background, line_color=text_color, linewidth=1)
    pitch.draw(ax=ax)
    ax.set_aspect('equal')
    
    if flipped==True:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    # Plotting those lines between players
    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y, passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                             lw=passes_between_df.width, color=text_color, zorder=2,alpha=0.5, ax=ax)

    # Plotting the player nodes
    for index, row in average_locs_and_count_df.iterrows():
        player_initials = get_initials(row['playerName'])
        pitch.scatter(row['pass_avg_x'], row['pass_avg_y'], s=row['marker_size'], marker='o', color=background, edgecolor=col,
                                    linewidth=2, alpha=1,zorder=2, ax=ax)
        pitch.annotate(player_initials, xy=(row.pass_avg_x, row.pass_avg_y), c=text_color, ha='center', va='center', size=18,zorder=2, ax=ax)

            

    dfteam = df[df['teamName'] == teamName]
    df_xT = dfteam[((dfteam['type'] == 'Pass') | (dfteam['type'] == 'Carry')) & (dfteam['xT'] > 0)]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [background, col])
    
    bin_statistic = pitch.bin_statistic(df_xT.x, df_xT.y, statistic='count', bins=(20, 20))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, edgecolors=background,alpha=0.5,zorder=1)

    pitch.annotate(f'Centralization Index: {ci:.2f}', xy=(0, -4), ha='left', va='center', fontsize=30, color=text_color, fontproperties=font_prop, ax=ax)
    
    return

def get_passing_stats(match_df, teamName):
    passes_df = match_df[(match_df['type'] == 'Pass') & (match_df['teamName'] == teamName)].copy()
    total_passes = passes_df[passes_df['teamName'] == teamName].shape[0]
    successful_passes = passes_df[(passes_df['teamName'] == teamName) & (passes_df['outcomeType'] == 'Successful')].shape[0]
    passing_accuracy = (successful_passes / total_passes) * 100
    final_third_passes = passes_df[(passes_df['teamName'] == teamName) & (passes_df['x'] < 75) & (passes_df['endX'] >= 75) & (passes_df['outcomeType'] == 'Successful')].shape[0]
    key_passes = passes_df[(passes_df['teamName'] == teamName) & (passes_df['passKey'] == True)].shape[0]
    crosses = passes_df[(passes_df['teamName'] == teamName) & (passes_df['qualifiers'].str.contains('Cross'))].shape[0]
    long_balls = passes_df[(passes_df['teamName'] == teamName) & (passes_df['qualifiers'].str.contains('Longball'))].shape[0]
    through_balls = passes_df[(passes_df['teamName'] == teamName) & (passes_df['qualifiers'].str.contains('Throughball'))].shape[0]
    progressive_passes = passes_df[(passes_df['teamName'] == teamName) & (passes_df['prog_pass']>=9.11) & (passes_df['x']>=35)].shape[0]
    pen_box_passes = passes_df[(passes_df['teamName'] == teamName) & (passes_df['endX']>=88.5) & (passes_df['endY']>=13.6) & (passes_df['endY']<=54.4)].shape[0]
    xT_by_Pass = passes_df['xT'].sum().round(3)

    result_df = pd.DataFrame({
        'Team': [teamName],
        'Total Passes': [int(total_passes)],
        'Passing Accuracy (%)': [round(passing_accuracy, 2)],
        'Final Third Entries': [int(final_third_passes)],
        'Key Passes': [int(key_passes)],
        'Crosses': [int(crosses)],
        'Long Balls': [int(long_balls)],
        'Through Balls': [int(through_balls)],
        'Progressive Passes': [int(progressive_passes)],
        'Pen Box Passes': [int(pen_box_passes)],
        'Expected Threat By Pass': [xT_by_Pass]
    })

    return result_df

def passmaps(ax,match_df,team,team_color,background,text_color,passtype):
    mask_passes = (match_df.type == 'Pass') & (match_df.teamName == team)
    team_passes_df = match_df.loc[mask_passes]

    pitch = VerticalPitch(pitch_type='uefa',half=True, corner_arcs=True, pitch_color=background,
                  line_zorder=2,line_color='white', linewidth=1)
    pitch.draw(ax=ax)
    ax.set_facecolor(background)
    if passtype == 'Final Third Entries':
        team_passes_df = team_passes_df[(team_passes_df['x'] < 75) & (team_passes_df['endX'] >= 75) & (team_passes_df['outcomeType'] == 'Successful')]
        for _,row in team_passes_df.iterrows():
            marker = '*' if row['passKey'] else 'o'
            size = 1000 if row['passKey'] else 200
            color = 'green' if row['assist'] else team_color
            linewidth = 5 if row['assist'] else 1
            pitch.lines(row.x, row.y, row.endX, row.endY,lw=linewidth, color=color, alpha=0.8, zorder=2, ax=ax)
            pitch.scatter(row.endX, row.endY,marker=marker, s=size, color=color, edgecolor=text_color, linewidth=2, zorder=3, ax=ax)

    elif passtype == 'Crosses':
        team_passes_df = team_passes_df[(team_passes_df['qualifiers'].str.contains('Cross')) & (team_passes_df['outcomeType'] == 'Successful')]
        for _,row in team_passes_df.iterrows():
            marker = '*' if row['passKey'] else 'o'
            size = 1000 if row['passKey'] else 200
            color = 'green' if row['assist'] else team_color
            linewidth = 5 if row['assist'] else 1
            pitch.lines(row.x, row.y, row.endX, row.endY,lw=linewidth, color=color, alpha=0.8, zorder=2, ax=ax)
            pitch.scatter(row.endX, row.endY,marker=marker, s=size, color=color, edgecolor=text_color, linewidth=2, zorder=3, ax=ax)
    
    elif passtype == 'Long Balls':
        team_passes_df = team_passes_df[(team_passes_df['qualifiers'].str.contains('Longball')) & (team_passes_df['outcomeType'] == 'Successful')]
        for _,row in team_passes_df.iterrows():
            marker = '*' if row['passKey'] else 'o'
            size = 1000 if row['passKey'] else 200
            color = 'green' if row['assist'] else team_color
            linewidth = 5 if row['assist'] else 1
            pitch.lines(row.x, row.y, row.endX, row.endY,lw=linewidth, color=color, alpha=0.8, zorder=2, ax=ax)
            pitch.scatter(row.endX, row.endY,marker=marker, s=size, color=color, edgecolor=text_color, linewidth=2, zorder=3, ax=ax)
    
    elif passtype == 'Through Balls':
        team_passes_df = team_passes_df[(team_passes_df['qualifiers'].str.contains('Throughball')) & (team_passes_df['outcomeType'] == 'Successful')]
        for _,row in team_passes_df.iterrows():
            marker = '*' if row['passKey'] else 'o'
            size = 1000 if row['passKey'] else 200
            color = 'green' if row['assist'] else team_color
            linewidth = 5 if row['assist'] else 1
            pitch.lines(row.x, row.y, row.endX, row.endY,lw=linewidth, color=color, alpha=0.8, zorder=2, ax=ax)
            pitch.scatter(row.endX, row.endY,marker=marker, s=size, color=color, edgecolor=text_color, linewidth=2, zorder=3, ax=ax)

    
    player_pass_counts = team_passes_df.groupby(['playerName']).size().reset_index(name='Count')
    top_passers = (player_pass_counts.sort_values(['Count'], ascending=[False]).head(5).reset_index(drop=True))
    
    return top_passers
    
def plot_donut_charts(ax, action_types, team_a_stats, team_b_stats, team1, team2,team1_facecolor, team2_facecolor):
    # Ensure the main axis background is black
    ax.set_facecolor('black')

    # Number of action types
    num_actions = len(action_types)
    
    # Loop through each action and create a new donut chart in a vertically stacked layout
    for i, action in enumerate(action_types):
        # Create a smaller inset axis for each donut, adjust its vertical position
        inset_ax = ax.inset_axes([0.1, 1 - (i+1) * 1.1 / num_actions, 1, 0.55])
        
        sizes = [team_a_stats[i], team_b_stats[i]]
        labels = [team_a_stats[i], team_b_stats[i]]
        colors = [team1_facecolor, team2_facecolor]  # Assign distinct colors to teams
        
        wedges, texts = inset_ax.pie(
            sizes, 
            labels=labels, 
            startangle=90, 
            colors=colors, 
            wedgeprops=dict(width=0.2),  # Adjust width for thicker donut
            textprops={'color': 'white', 'fontsize': 12}
        )
        
        # Add a circle for the donut hole
        circle = plt.Circle((0, 0), 0.7, color='black', fc='black')
        inset_ax.add_artist(circle)
        
        # Add action label at the center
        inset_ax.text(0, 0, action, horizontalalignment='center', verticalalignment='center', 
                      fontsize=12, fontweight='bold', color='white',fontproperties=font_prop)
        
        # Set equal aspect ratio and hide the axes
        inset_ax.axis('equal')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    ax.axis('off')

def plot_duels_by_type(ax, df, team1_name, team2_name, duel_type,
                       team1_color, team2_color, background, text_color):

    # Define duel type filters
    offensive_types = ['TakeOn', 'GoodSkill', 'ShieldBallOpp']
    defensive_types = ['Tackle', 'Challenge', 'Dispossessed']

    # Filter data by duel type
    if duel_type == 'Total':
        df_duel = df[
            df['type'].isin(offensive_types + defensive_types) |
            (df['duelAerialWon'] == True) |
            (df['duelAerialLost'] == True)
        ]
    elif duel_type == 'Offensive':
        df_duel = df[df['type'].isin(offensive_types)]
    elif duel_type == 'Defensive':
        df_duel = df[df['type'].isin(defensive_types)]
    elif duel_type == 'Aerial':
        df_duel = df[(df['duelAerialWon'] == True) | (df['duelAerialLost'] == True)]
    else:
        raise ValueError("Invalid duel type selected.")

    # Create team-specific dataframes
    df1 = df_duel[(df_duel['teamName'] == team1_name) & (df_duel['outcomeType'] == 'Successful')].copy()
    df2 = df_duel[(df_duel['teamName'] == team2_name) & (df_duel['outcomeType'] == 'Successful')].copy()

    # Flip team2 coordinates to match attacking direction
    df2['x'] = 105 - df2['x']
    df1['y'] = 68 - df1['y']

    # Setup pitch
    pitch = Pitch(pitch_type='uefa', pitch_color=background, line_color=text_color,
                  linewidth=2, line_zorder=3)
    pitch.draw(ax=ax)

    # Bin settings
    bins_x, bins_y = 6, 5
    bin_w, bin_h = 105 / bins_x, 68 / bins_y

    # Bin duels by zone
    stat1 = pitch.bin_statistic(df1['x'], df1['y'], statistic='count', bins=(bins_x, bins_y))['statistic']
    stat2 = pitch.bin_statistic(df2['x'], df2['y'], statistic='count', bins=(bins_x, bins_y))['statistic']

    # Draw duel bars in each bin
    for j in range(bins_y):  # row (y)
        for i in range(bins_x):  # column (x)
            x0 = i * bin_w
            y0 = j * bin_h
            t1 = stat1[j, i]
            t2 = stat2[j, i]
            total = t1 + t2
            if total == 0:
                continue
            r1 = t1 / total
            r2 = t2 / total

            # Draw rectangles
            ax.add_patch(Rectangle((x0, y0), bin_w * r1, bin_h, color=team1_color, alpha=0.7, zorder=3))
            ax.add_patch(Rectangle((x0 + bin_w * r1, y0), bin_w * r2, bin_h, color=team2_color, alpha=0.7, zorder=3))

            # Text inside bins
            ax.text(x0 + bin_w/2, y0 + bin_h/2, f"{int(t1)}/{int(t2)}",
                    ha='center', va='center', fontsize=18, color=text_color, fontproperties=font_prop, zorder=4)

    # Direction arrows
    ax.text(0,  -3, 'Attacking Direction--->', color=team1_color, fontsize=22, ha='left', va='center', fontproperties=font_prop)
    ax.text(105, -3, '<---Attacking Direction', color=team2_color, fontsize=22, ha='right', va='center', fontproperties=font_prop)

    # Dashed grid overlay
    for i in range(1, bins_x):
        ax.vlines(i * bin_w, ymin=0, ymax=68, color=background, lw=2, ls='--', zorder=5)
    for j in range(1, bins_y):
        ax.hlines(j * bin_h, xmin=0, xmax=105, color=background, lw=2, ls='--', zorder=5)

    # Get top 5 duel winners for each team
    if 'playerName' in df_duel.columns:
        top_team1 = (
            df1.groupby('playerName')
            .size()
            .reset_index(name='DuelsWon')
            .sort_values(by='DuelsWon', ascending=False)
            .head(5)
        )

        top_team2 = (
            df2.groupby('playerName')
            .size()
            .reset_index(name='DuelsWon')
            .sort_values(by='DuelsWon', ascending=False)
            .head(5)
        )
    else:
        top_team1 = pd.DataFrame(columns=['HomePlayerName', 'HomeDuelsWon'])
        top_team2 = pd.DataFrame(columns=['AwayPlayerName', 'AwayDuelsWon'])

    return top_team1, top_team2

def get_defensive_action_df(df):
    # filter only defensive actions
    defensive_actions_ids = df.index[(df['type'] == 'Aerial') & (df['qualifiers'].str.contains('Defensive')) |
                                     (df['type'] == 'BallRecovery') |
                                     (df['type'] == 'BlockedPass') |
                                     (df['type'] == 'Challenge') |
                                     (df['type'] == 'Clearance') |
                                     (df['type'] == 'Error') |
                                     (df['type'] == 'Foul') |
                                     (df['type'] == 'Interception') |
                                     (df['type'] == 'Tackle')]
    df_defensive_actions = df.loc[defensive_actions_ids, ["index", "x", "y", "teamName", "playerId", "type", "outcomeType","name"]]

    return df_defensive_actions

def calculate_event_types(dataframe, event_types=None):
    # Group by 'teamName' and 'type', then count occurrences
    event_counts = dataframe.groupby(['teamName', 'type']).size().reset_index(name='count')
    
    # If event_types is provided, filter the event_counts DataFrame
    if event_types is not None:
        event_counts = event_counts[event_counts['type'].isin(event_types)]
    
    # Create a dictionary to hold the results
    results = {}
    
    # Populate the dictionary with team names and their event types and counts
    for team in event_counts['teamName'].unique():
        team_data = event_counts[event_counts['teamName'] == team]
        results[team] = {
            'types': team_data['type'].tolist(),
            'counts': team_data['count'].tolist()
        }
    
    return results

def get_da_count_df(team_name, defensive_actions_df, players_df):
    defensive_actions_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
    # add column with first eleven players only
    defensive_actions_df = defensive_actions_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')
    # calculate mean positions for players
    average_locs_and_count_df = (defensive_actions_df.groupby('playerId').agg({'x': ['median'], 'y': ['median', 'count']}))
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    #average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position', 'isFirstEleven']], on='playerId', how='left')
    average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')

    return  average_locs_and_count_df

def defensive_block(ax,df,hteam,average_locs_and_count_df, team_name, col,flipped=True):
    defensive_actions_df = get_defensive_action_df(df)
    defensive_actions_team_df = defensive_actions_df[defensive_actions_df["teamName"] == team_name]
    pitch = Pitch(pitch_type='uefa', pitch_color=background, line_color='white', linewidth=1.5, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_facecolor(background)
    if team_name == hteam:
        ax.text(32,73,"Defensive Actions", color='white', fontsize=25,fontproperties=font_prop)
    else:
        ax.text(75,-5,"Defensive Actions", color='white', fontsize=25,fontproperties=font_prop)
    #ax.set_xlim(-0.5, 105.5)
    #ax.set_ylim(-0.5, 68.5)
    color = np.array(to_rgba(col))
    flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", [background, col], N=500)
    kde = pitch.kdeplot(defensive_actions_team_df.x, defensive_actions_team_df.y, ax=ax, fill=True, levels=5000,alpha=0.7, thresh=0.02, cut=4, cmap=flamingo_cmap)

    average_locs_and_count_df = average_locs_and_count_df.reset_index(drop=True)
    for index,row in defensive_actions_team_df.iterrows():
        if row['type'] == 'Aerial':
            pitch.scatter(row.x, row.y, s=20, marker='x', color='white', alpha=0.8, ax=ax)
        elif row['type'] == 'BallRecovery':
            pitch.scatter(row.x, row.y, s=20, marker='o', color='white', alpha=0.8, ax=ax)
        elif row['type'] == 'Challenge':
            pitch.scatter(row.x, row.y, s=20, marker='^', color='white', alpha=0.8, ax=ax)
        elif row['type'] == 'Interception':
            pitch.scatter(row.x, row.y, s=20, marker='+', color='white', alpha=0.8, ax=ax)
        elif row['type'] == 'Tackle':
            pitch.scatter(row.x, row.y, s=20, marker='*', color='white', alpha=0.8, ax=ax)
        else:
            pitch.scatter(row.x, row.y, s=10, marker='.', color='white', alpha=0.5, ax=ax)

    if flipped == True:
        ax.invert_xaxis()
        ax.invert_yaxis()

def plot_donut_charts_def(ax, action_types, team_a_stats, team_b_stats, team1, team2,team1_facecolor,team2_facecolor):
    # Ensure the main axis background is black
    ax.set_facecolor(background)

    # Number of action types
    num_actions = len(action_types)
    
    # Loop through each action and create a new donut chart in a vertically stacked layout
    for i, action in enumerate(action_types):
        # Create a smaller inset axis for each donut, adjust its vertical position
        inset_ax = ax.inset_axes([-0.05, 1 - (i+1) * 1.2 / num_actions, 1.1, 0.65])
        
        sizes = [team_a_stats[i], team_b_stats[i]]
        labels = [team_a_stats[i], team_b_stats[i]]
        colors = [team1_facecolor, team2_facecolor]  # Assign distinct colors to teams
        
        wedges, texts = inset_ax.pie(
            sizes, 
            labels=labels, 
            startangle=90, 
            colors=colors, 
            wedgeprops=dict(width=0.2),  # Adjust width for thicker donut
            textprops={'color': 'white', 'fontsize': 15}
        )
        
        # Add a circle for the donut hole
        circle = plt.Circle((0, 0), 0.7, color='black', fc='black')
        inset_ax.add_artist(circle)
        
        inset_ax.text(0, 0, action, horizontalalignment='center', verticalalignment='center', 
                      fontsize=15, color='white',fontproperties=font_prop)
        
        # Set equal aspect ratio and hide the axes
        inset_ax.axis('equal')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])

    ax.axis('off')

def xT_momemtum(ax,df,team1_name,team2_name,team1_facecolor,team2_facecolor):
    xT_data = df[((df['type'] == 'Pass') | (df['type'] == 'Carry')) & (df['outcomeType'] == 'Successful')]
    xT_data = xT_data[['xT', 'minute', 'second', 'teamName', 'type']]
    xT_data['xT_clipped'] = np.clip(xT_data['xT'], 0, 0.1)

    max_xT_per_minute = xT_data.groupby(['teamName', 'minute'])['xT_clipped'].max().reset_index()

    minutes = sorted(xT_data['minute'].unique())
    weighted_xT_sum = {
        team1_name: [],
        team2_name: []
    }
    momentum = []
    
    window_size = 3
    decay_rate = 0.1
    
    
    for current_minute in minutes:
        for team in weighted_xT_sum.keys():
            
            recent_xT_values = max_xT_per_minute[
                                                (max_xT_per_minute['teamName'] == team) & 
                                                (max_xT_per_minute['minute'] <= current_minute) & 
                                                (max_xT_per_minute['minute'] > current_minute - window_size)
                                            ]
            
            weights = np.exp(-decay_rate * (current_minute - recent_xT_values['minute'].values))
            weighted_sum = np.sum(weights * recent_xT_values['xT_clipped'].values)
            weighted_xT_sum[team].append(weighted_sum)
    
        momentum.append(weighted_xT_sum[team1_name][-1] - weighted_xT_sum[team2_name][-1])
    
    momentum_df = pd.DataFrame({
        'minute': minutes,
        'momentum': momentum
    })

    ax.set_facecolor(background)

    ax.tick_params(axis='x', colors='white')
    ax.margins(x=0)
    ax.set_xticks([0,15,30,45,60,75,90])
    
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.set_ylim(-0.15, 0.2)
    
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    momentum_df['smoothed_momentum'] = gaussian_filter1d(momentum_df['momentum'], sigma=0.2)
    ax.plot(momentum_df['minute'], momentum_df['smoothed_momentum'], color='white')
    
    ax.axhline(0, color='white', linestyle='--', linewidth=0.5)
    ax.fill_between(momentum_df['minute'], momentum_df['smoothed_momentum'], where=(momentum_df['smoothed_momentum'] > 0), color=team1_facecolor, alpha=0.9, interpolate=True)
    ax.fill_between(momentum_df['minute'], momentum_df['smoothed_momentum'], where=(momentum_df['smoothed_momentum'] < 0), color=team2_facecolor, alpha=0.9, interpolate=True) 
    
    
    #scores = df[df['shot_outcome'] == 'Goal'].groupby('team')['shot_outcome'].count().reindex(set(df['team']), fill_value=0)
    #ax.set_xlabel('Minute', color='white', fontsize=15, fontweight='bold', fontproperties=font_prop)
    ax.set_ylabel('Momentum', color='white', fontsize=15, fontweight='bold', fontproperties=font_prop)
    ax.text(30,0.2,f'xT Momentum Flow', color='white', fontsize=25, fontproperties=font_prop)
    
    #home_team_text = ax.text(9, 0.1, team1_name, fontsize=15, ha='center', fontproperties=font_prop, fontweight='bold', color='white')
    #home_team_text.set_bbox(dict(facecolor=team1_facecolor, alpha=0.5, edgecolor='white', boxstyle='round'))
    #away_team_text = ax.text(7, -0.084, team2_name, fontsize=15, ha='center', fontproperties=font_prop, fontweight='bold', color='white')
    #away_team_text.set_bbox(dict(facecolor=team2_facecolor, alpha=0.5, edgecolor='white', boxstyle='round'))
    
    
    goals = df[df['type']=='Goal'][['minute', 'teamName','isOwnGoal_x']]
    goals.loc[(goals['teamName'] == team1_name) & (goals['isOwnGoal_x'] == True), 'teamName'] = team2_name
    goals.loc[(goals['teamName'] == team2_name) & (goals['isOwnGoal_x'] == True), 'teamName'] = team1_name
    for _, row in goals.iterrows():
        ymin, ymax = (0.4, 0.85) if row['teamName'] == team1_name else (0.08, 0.43)
        ax.axvline(row['minute'], color='white', linestyle='--', linewidth=1.5, alpha=0.5, ymin=ymin, ymax=ymax)
        ax.scatter(row['minute'], (1 if row['teamName'] == team1_name else -0.85)*0.15, color='white', s=200, zorder=10, alpha=0.9)
        ax.text(row['minute'], (1.2 if row['teamName'] == team1_name else -1)*0.14, 'G', fontsize=15, ha='center', va='center', fontproperties=font_prop, color=team1_facecolor if row['teamName'] == team1_name else team2_facecolor)

def Final_third_entry(ax,df,hteam,ateam,team1_facecolor,team2_facecolor, team_name, col):
    # Final third Entry means passes or carries which has started outside the Final third and ended inside the final third
    dfpass = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['x']<75) & (df['endX']>=75) & (df['outcomeType']=='Successful') &
                (~df['qualifiers'].str.contains('Freekick'))]
    dfcarry = df[(df['teamName']==team_name) & (df['type']=='Carry') & (df['x']<75) & (df['endX']>=75)]
    pitch = Pitch(pitch_type='uefa', pitch_color=background, line_color='white', linewidth=1,
                          corner_arcs=True)
    pitch.draw(ax=ax)
    ax.set_xlim(-0.5, 105.5)
    # ax.set_ylim(-0.5, 68.5)

    if team_name == ateam:
        ax.invert_xaxis()
        ax.invert_yaxis()

    pass_count = len(dfpass) + len(dfcarry)

    # calculating the counts
    left_entry = len(dfpass[dfpass['y']>=45.33]) + len(dfcarry[dfcarry['y']>=45.33])
    mid_entry = len(dfpass[(dfpass['y']>=22.67) & (dfpass['y']<45.33)]) + len(dfcarry[(dfcarry['y']>=22.67) & (dfcarry['y']<45.33)])
    right_entry = len(dfpass[(dfpass['y']>=0) & (dfpass['y']<22.67)]) + len(dfcarry[(dfcarry['y']>=0) & (dfcarry['y']<22.67)])
    left_percentage = round((left_entry/pass_count)*100)
    mid_percentage = round((mid_entry/pass_count)*100)
    right_percentage = round((right_entry/pass_count)*100)

    ax.hlines(22.67, xmin=0, xmax=70, colors='white', linestyle='dashed', alpha=0.35)
    ax.hlines(45.33, xmin=0, xmax=70, colors='white', linestyle='dashed', alpha=0.35)
    ax.vlines(70, ymin=-2, ymax=70, colors='white', linestyle='dashed', alpha=0.55)

    # showing the texts in the pitch
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=background, alpha=0.75)
    if col == team1_facecolor:
        ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=team1_facecolor, fontsize=24, va='center', ha='center', bbox=bbox_props,fontproperties=font_prop)
        ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=team1_facecolor, fontsize=24, va='center', ha='center', bbox=bbox_props,fontproperties=font_prop)
        ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=team1_facecolor, fontsize=24, va='center', ha='center', bbox=bbox_props,fontproperties=font_prop)
    else:
        ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=team2_facecolor, fontsize=24, va='center', ha='center', bbox=bbox_props,fontproperties=font_prop)
        ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=team2_facecolor, fontsize=24, va='center', ha='center', bbox=bbox_props,fontproperties=font_prop)
        ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=team2_facecolor, fontsize=24, va='center', ha='center', bbox=bbox_props,fontproperties=font_prop)

    # plotting the passes
    pro_pass = pitch.lines(dfpass.x, dfpass.y, dfpass.endX, dfpass.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
    # plotting some scatters at the end of each pass
    pro_pass_end = pitch.scatter(dfpass.endX, dfpass.endY, s=35, edgecolor=col, linewidth=1, color=background, zorder=2, ax=ax)
    # plotting carries
    for index, row in dfcarry.iterrows():
        arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=20,
                                        alpha=1, linewidth=2, linestyle='--')
        ax.add_patch(arrow)

    counttext = f"{pass_count} Final Third Entries"

    # Heading and other texts
    if col == team1_facecolor:
        ax.text(50,80,f"{counttext}", color='white', fontsize=25, fontweight='bold',ha='center', va='center', path_effects=path_eff,fontproperties=font_prop)
        ax.text(87.5, 70, '<------------ Final third ------------>', color='white', ha='center', va='center',fontproperties=font_prop)
        pitch.lines(53, -2, 73, -2, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
        ax.scatter(73,-2, s=35, edgecolor=col, linewidth=1, color=background, zorder=2)
        arrow = patches.FancyArrowPatch((83, -2), (103, -2), arrowstyle='->', color=col, zorder=4, mutation_scale=20,
                                        alpha=1, linewidth=2, linestyle='--')
        ax.add_patch(arrow)
        ax.text(63, -5, f'Entry by Pass: {len(dfpass)}', fontsize=15, color='white', ha='center', va='center',fontproperties=font_prop)
        ax.text(93, -5, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color='white', ha='center', va='center',fontproperties=font_prop)

    else:
        ax.text(50,-12,f"{counttext}", color='white', fontsize=25, fontweight='bold',ha='center', va='center', path_effects=path_eff,fontproperties=font_prop)
        ax.text(87.5, -2, '<---------- Final third ---------->', color='white', ha='center', va='center',fontproperties=font_prop)
        pitch.lines(53, 70, 73, 70, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
        ax.scatter(73,70, s=35, edgecolor=col, linewidth=1, color=background, zorder=2)
        arrow = patches.FancyArrowPatch((83, 70), (103, 70), arrowstyle='->', color=col, zorder=4, mutation_scale=20,
                                        alpha=1, linewidth=2, linestyle='--')
        ax.add_patch(arrow)
        ax.text(63, 73, f'Entry by Pass: {len(dfpass)}', fontsize=15, color='white', ha='center', va='center',fontproperties=font_prop)
        ax.text(93, 73, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color='white', ha='center', va='center',fontproperties=font_prop)

    return {
        'Team_Name': team_name,
        'Total_Final_Third_Entries': pass_count,
        'Final_Third_Entries_From_Left': left_entry,
        'Final_Third_Entries_From_Center': mid_entry,
        'Final_Third_Entries_From_Right': right_entry,
        'Entry_By_Pass': len(dfpass),
        'Entry_By_Carry': len(dfcarry)
    }

def plot_ppda(df, flag, ax,team1_name,team2_name,team1_facecolor,team2_facecolor):
    df_ppda = df.copy()
    goals = df[df['type'] == 'Goal'][['minute', 'second', 'teamName']]
    #goals['timestamp'] = goals['minute'] * 60 + df_ppda['second']
    #goals['time_bin'] = (goals['timestamp'] // 900).astype(int)
    
    # Step 1: Add a 'timestamp' column for easier time calculation
    df_ppda['timestamp'] = df_ppda['minute'] * 60 + df_ppda['second']
    
    # Step 2: Identify possessions
    if flag:
        df_ppda['possession_change'] = (df_ppda['teamName'] != df_ppda['teamName'].shift(1)) | (
            df_ppda['type'].isin(['Interception', 'Tackle', 'Clearance', 'BlockedPass','BallRecovery','Challenge']) &
            (df_ppda['outcomeType'] == 'Successful')
        )
    else:
        df_ppda['possession_change'] = (df_ppda['teamName'] != df_ppda['teamName'].shift(1)) | (
            df_ppda['type'].isin(['Interception', 'Tackle', 'Clearance', 'BlockedPass','BallRecovery','Challenge'])
        )

    df_ppda['possession_id'] = df_ppda['possession_change'].cumsum()
    
    df_ppda['timestamp'] = pd.to_numeric(df_ppda['timestamp'], errors='coerce')  # Convert non-numeric to NaN
    df_ppda = df_ppda.dropna(subset=['timestamp'])
    
    # Step 3: Aggregate possession data
    if flag:
        possessions = df_ppda.groupby('possession_id').agg(
            team=('teamName', 'first'),
            opposition=('oppositionTeamName', 'first'),
            passes=('type', lambda x: x.isin(['Pass']).sum()),
            defensive_actions=('type', 
                               lambda x: ((x.isin(['Interception', 'Tackle', 'Clearance', 'BlockedPass','BallRecovery','Challenge'])) & 
                                          (df.loc[x.index, 'outcomeType'] == 'Successful')).sum())
        ).reset_index()
    else:
        possessions = df_ppda.groupby('possession_id').agg(
            team=('teamName', 'first'),
            opposition=('oppositionTeamName', 'first'),
            passes=('type', lambda x: x.isin(['Pass']).sum()),
            defensive_actions=('type', 
                               lambda x: ((x.isin(['Interception', 'Tackle', 'Clearance', 'BlockedPass','BallRecovery','Challenge']))).sum())
        ).reset_index()

    # Step 4: Add time bins (e.g., 15-minute windows)
    df_ppda['time_bin'] = (df_ppda['timestamp'] // 900).astype(int)  # 900 seconds = 15 minutes
    
    # Step 5: Merge possession data with time bins
    possessions = possessions.merge(
        df_ppda[['possession_id', 'time_bin']].drop_duplicates(),
        on='possession_id',
        how='left'
    )
    
    # Step 6: Calculate PPDA for each time bin
    ppda = possessions.groupby(['time_bin', 'opposition']).agg(
        total_passes=('passes', 'sum'),
        total_def_actions=('defensive_actions', 'sum')
    ).reset_index()
    
    ppda['PPDA'] = ppda['total_passes'] / ppda['total_def_actions']
    ppda['PPDA'] = ppda['PPDA'].fillna(float('inf'))  # Handle cases with no defensive actions
    
    # Data preparation
    teams = ppda['opposition'].unique()  # List of teams
    
    ax.set_facecolor(background)
    
    # Plot PPDA values for each team
    for team in teams:
        team_data = ppda[ppda['opposition'] == team]
        if team == team1_name:
            ax.plot(team_data['time_bin'], team_data['PPDA'], label=team, marker='o', color=team1_facecolor, linewidth=3)
        else:
            ax.plot(team_data['time_bin'], team_data['PPDA'], label=team, marker='o', color=team2_facecolor, linewidth=3)
        
        # Annotate PPDA values on the plot
        for _, row in team_data.iterrows():
            ax.text(row['time_bin'], row['PPDA'] + 1.2, f"{row['PPDA']:.1f}", fontsize=12, ha='center', va='bottom', color='white')
    
    # Adjust Y-axis: reverse and set interval
    ax.invert_yaxis()  # Reverse the y-axis
    #ax.set_yticks(range(0, 21, 5))  # Set y-axis ticks and color
    ax.set_xticks(ppda['time_bin'].unique())  # Set x-axis ticks and color
    
    # Customize spines
    #ax.spines['top'].set_color('black')
    #ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    
    # Chart formatting
    ax.set_title('Pressing Intensity - PPDA', fontproperties=font_prop, fontsize=25, color='white', y=1.05)
    ax.set_xlabel('Time Bin (15-minute intervals)', fontsize=15, color='white', fontproperties=font_prop)
    ax.set_ylabel('PPDA Value', fontsize=15, color='white', fontproperties=font_prop)
    #ax.legend(title='Teams', facecolor='white', edgecolor='white', labelcolor='black', title_fontsize=10)

    # Add grid with low opacity for better visibility
    ax.grid(alpha=0.5, color='grey')

    # Add additional text below the plot using the figure's figtext
    #ax.figure.figtext(0.55, -0.05, "PPDA - Passes (Allowed) Per Defensive Action", wrap=True, 
    #                  fontproperties=font_prop, horizontalalignment='center', fontsize=15, color='white')
    #ax.figure.figtext(0.55, -0.1, "Lower Value Represents Higher Pressing Intensity", wrap=True, 
    #                 fontproperties=font_prop, horizontalalignment='center', fontsize=15, color='white')

    # Tight layout and show the chart
    plt.tight_layout()

def plot_fieldtilt(df, ax,team1_name,team2_name,team1_facecolor,team2_facecolor):
    ## Ratio of each team’s final third touches compared to the total final third touches
    df_final_third = df[(df['x'] > 75) & (df['isTouch'] == 1) & (~df['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))]
    df_final_third['time_bin'] = np.floor(df_final_third['minute'] / 15).astype(int)
    
    final_third_touches = df_final_third.groupby(['time_bin', 'teamName']).size().reset_index(name='final_third_touches')
    total_touches = df_final_third.groupby('time_bin').size().reset_index(name='total_final_third_touches')
    
    field_tilt_data = final_third_touches.merge(total_touches, on='time_bin')
    field_tilt_data['Field Tilt'] = field_tilt_data['final_third_touches'] / field_tilt_data['total_final_third_touches']
    field_tilt_data = field_tilt_data[['time_bin', 'teamName', 'Field Tilt']]
    
    teams = field_tilt_data['teamName'].unique()  # List of teams
    
    # Set background color
    ax.set_facecolor(background)
    
    for team in teams:
        team_data = field_tilt_data[field_tilt_data['teamName'] == team]
        if team == team1_name:
            ax.plot(team_data['time_bin'], team_data['Field Tilt'], label=team, marker='o', color=team1_facecolor, linewidth=3)
        else:
            ax.plot(team_data['time_bin'], team_data['Field Tilt'], label=team, marker='o', color=team2_facecolor, linewidth=3)
        
        for _, row in team_data.iterrows():
            ax.text(row['time_bin'], row['Field Tilt'] + 0.03, f"{row['Field Tilt']:.2f}", fontsize=12, ha='center', va='bottom', color='white')
    
    # Adjust Y-axis
    #ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticks(field_tilt_data['time_bin'].unique())
    
    # Set tick colors
    #ax.tick_params(axis='both', colors='white')
    
    # Customize spines
    #ax.spines['top'].set_color('black')
    #ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    
    # Chart formatting
    ax.set_title('Field Tilt', fontproperties=font_prop, fontsize=25, color='white', y=1.05)
    ax.set_xlabel('Time Bin (15-minute intervals)', fontsize=15, color='white', fontproperties=font_prop)
    ax.set_ylabel('Tilt Value', fontsize=15, color='white', fontproperties=font_prop)
    #ax.legend(title='Teams', facecolor='white', edgecolor='white', labelcolor='black', title_fontsize=10)
    
    # Add grid
    ax.grid(alpha=0.5, color='grey')
    
    # Add additional text below the plot
    #plt.figtext(0.5, -0.05, "Ratio of each team’s final third touches compared to the total final third touches", 
    #            wrap=True, fontproperties=font_prop, horizontalalignment='center', fontsize=15, color='white')

def get_next_events_by_index(df, current_index, num_events, target_types=None):
    """
    Get the next consecutive rows based on DataFrame indices and filter by event type.
    Only process if the current event is of type 'Pass'.
    """
    # Check if the current event is a 'Pass'
    if df.loc[current_index, 'type'] != 'Pass':
        return []  # Return an empty list if the type is not 'Pass'

    # Get the valid range of indices for the next events
    max_index = len(df) - 1
    next_indices = [i for i in range(current_index + 1, current_index + 1 + num_events) if i <= max_index]
    
    # Fetch the rows using the valid indices
    next_events = df.iloc[next_indices].reset_index(drop=True)
    
    # Filter for relevant types, if provided
    if target_types:
        next_events = next_events[next_events['type'].isin(target_types)]
    
    return next_events.to_dict('records')

def extract_last_shot_coordinates(next_events):
    if next_events:
        # Filter out events that are shots (Shot types: 'SavedShot', 'MissedShot', 'ShotOnPost', 'Goal')
        shot_events = [event for event in next_events if event['type'] in ['SavedShot', 'MissedShots', 'ShotOnPost', 'Goal']]
        if shot_events:
            # Get the coordinates of the last shot event
            last_shot = shot_events[-1]
            return last_shot['x'], last_shot['y']
    return None, None  # Return None if no shots found

def plot_corners(corners,team1,team2,team1_color,team2_color):
    corners_1 = corners[(corners['side'] == 'Left') & (corners['teamName'] == team1)].copy()
    corners_2 = corners[(corners['side'] == 'Right')  & (corners['teamName'] == team1)].copy()
    corners_3 = corners[(corners['side'] == 'Left') & (corners['teamName'] == team2)].copy()
    corners_4 = corners[(corners['side'] == 'Right')  & (corners['teamName'] == team2)].copy()

    def process_corners(df):
        if df.empty:
            return pd.DataFrame({'x':[0],'endX':[0],'endY':[0], 'y':[0],'next_shot_x':[0],'next_shot_y':[0],'total_shots': [0], 'goals': [0], 'shots_ot': [0]}, index=pd.Index(['Empty'], name='type'))
        
        # Calculate the number of shots (next_events_type is not None)
        df['total_shots'] = df['next_events_type'].apply(
            lambda x: 1 if x is not None else 0
        )
        df['goals'] = df['next_events_type'].apply(
            lambda x: 1 if 'Goal' in (x if isinstance(x, list) else [x]) else 0
        )
        #df['shots_ot'] = df['next_events_type'].apply(
        #    lambda x: 1 if 'SavedShot' in (x if isinstance(x, list) else [x]) else 0
        #)
        df['shots_ot'] = df['next_events_type'].apply(
            lambda x: 1 if any(event in (x if isinstance(x, list) else [x]) for event in ['SavedShot', 'Goal']) else 0
        )

        
        return df

    # Process both sides
    corners_1 = process_corners(corners_1)
    corners_2 = process_corners(corners_2)
    corners_3 = process_corners(corners_3)
    corners_4 = process_corners(corners_4)

    # Now you can calculate the total number of shots and goals by corner type
    shots_by_corner_type_1 = corners_1.groupby('type')['total_shots'].sum()
    goals_by_corner_type_1 = corners_1.groupby('type')['goals'].sum()
    shots_ot_by_corner_type_1 = corners_1.groupby('type')['shots_ot'].sum()
    
    shots_by_corner_type_2 = corners_2.groupby('type')['total_shots'].sum()
    goals_by_corner_type_2 = corners_2.groupby('type')['goals'].sum()
    shots_ot_by_corner_type_2 = corners_2.groupby('type')['shots_ot'].sum()

    shots_by_corner_type_3 = corners_3.groupby('type')['total_shots'].sum()
    goals_by_corner_type_3 = corners_3.groupby('type')['goals'].sum()
    shots_ot_by_corner_type_3 = corners_3.groupby('type')['shots_ot'].sum()

    shots_by_corner_type_4 = corners_4.groupby('type')['total_shots'].sum()
    goals_by_corner_type_4 = corners_4.groupby('type')['goals'].sum()
    shots_ot_by_corner_type_4 = corners_4.groupby('type')['shots_ot'].sum()
    
    # Combine them to get a complete picture
    corner_summary_1 = pd.DataFrame({
        'total_shots': shots_by_corner_type_1,
        'goals': goals_by_corner_type_1,
        'shots_ot': shots_ot_by_corner_type_1
    })
    
    corner_summary_2 = pd.DataFrame({
        'total_shots': shots_by_corner_type_2,
        'goals': goals_by_corner_type_2,
        'shots_ot': shots_ot_by_corner_type_2
    })

    corner_summary_3 = pd.DataFrame({
        'total_shots': shots_by_corner_type_3,
        'goals': goals_by_corner_type_3,
        'shots_ot': shots_ot_by_corner_type_3
    })

    corner_summary_4 = pd.DataFrame({
        'total_shots': shots_by_corner_type_4,
        'goals': goals_by_corner_type_4,
        'shots_ot': shots_ot_by_corner_type_4
    })

    FIGWIDTH = 12
    FIGHEIGHT = 9
    NROWS = 2
    NCOLS = 2
    SPACE = 0
    MAX_GRID = 0.95
    
    pitch = VerticalPitch(pitch_type='uefa',pad_bottom=-15,line_color='white',goal_type='box', linewidth=0.5, half=True,
                          pitch_color=background)
    
    GRID_WIDTH, GRID_HEIGHT = pitch.grid_dimensions(figwidth=FIGWIDTH, figheight=FIGHEIGHT,
                                                    nrows=NROWS, ncols=NCOLS,
                                                    max_grid=MAX_GRID, space=SPACE)
    TITLE_HEIGHT = 0.01
    ENDNOTE_HEIGHT = 0
    
    fig, ax = pitch.grid(figheight=FIGHEIGHT, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT,
                         space=SPACE, ncols=NCOLS, nrows=NROWS, title_height=TITLE_HEIGHT,
                         endnote_height=ENDNOTE_HEIGHT, axis=False)
    fig.set_facecolor(background)
    fig.suptitle(f'Corner Analysis', fontsize=30,fontproperties=font_prop, color='white', y=0.92)
    img_team1 = mpimg.imread(f'C:/Users/acer/Documents/GitHub/IndianCitizen/ScorePredict/Images/TeamLogos/{team1}.png')
    ax_image = add_image(
        img_team1, fig, left=0.2, bottom=0.85, width=0.12, height=0.12,aspect='equal'
    )

    img_team2 = mpimg.imread(f'C:/Users/acer/Documents/GitHub/IndianCitizen/ScorePredict/Images/TeamLogos/{team2}.png')
    ax_image = add_image(
        img_team2, fig, left=0.68, bottom=0.85, width=0.12, height=0.12,aspect='equal'
    )

    if not corners_1.empty:
        pitch.arrows(corners_1.x, corners_1.y,
                     corners_1.endX, corners_1.endY, width=1.5,alpha=0.8,zorder=3,
                     headwidth=8, headlength=5, color='white', ax=ax['pitch'][0,0], label='Corners From Left')
        pitch.scatter(corners_1.next_shot_x, corners_1.next_shot_y,s=500,marker='p',color=team1_color,zorder=2,alpha=0.9, ax=ax['pitch'][0,0], edgecolors='white')

    if not corners_2.empty:
        pitch.arrows(corners_2.x, corners_2.y,
                     corners_2.endX, corners_2.endY, width=1.5,alpha=0.8,zorder=3,
                     headwidth=8, headlength=5, color='white', ax=ax['pitch'][1,0], label='Corners From Right')
        pitch.scatter(corners_2.next_shot_x, corners_2.next_shot_y,s=500,marker='p',color=team1_color,zorder=2,alpha=0.9, ax=ax['pitch'][1,0], edgecolors='white')

    pitch.scatter(80,45,color=team1_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][0,0])
    pitch.annotate(corner_summary_1['total_shots'][0],xy = (80,45),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,0])
    pitch.annotate('Shots',xy = (75,45),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,0])
    
    pitch.scatter(80,34,color=team1_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][0,0])
    pitch.annotate(corner_summary_1['shots_ot'][0],xy = (80,34),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,0])
    pitch.annotate('Shots(OT)',xy = (75,34),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,0])
    
    pitch.scatter(80,23,color=team1_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][0,0])
    pitch.annotate(corner_summary_1['goals'][0],xy = (80,23),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,0])
    pitch.annotate('Goals',xy = (75,23),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,0])
    
    
    pitch.scatter(80,45,color=team1_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][1,0])
    pitch.annotate(corner_summary_2['total_shots'][0],xy = (80,45),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,0])
    pitch.annotate('Shots',xy = (75,45),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,0])
    
    pitch.scatter(80,34,color=team1_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][1,0])
    pitch.annotate(corner_summary_2['shots_ot'][0],xy = (80,34),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,0])
    pitch.annotate('Shots(OT)',xy = (75,34),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,0])
    
    
    pitch.scatter(80,23,color=team1_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][1,0])
    pitch.annotate(corner_summary_2['goals'][0],xy = (80,23),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,0])
    pitch.annotate('Goals',xy = (75,23),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,0])

    if not corners_3.empty:
        pitch.arrows(corners_3.x, corners_3.y,
                     corners_3.endX, corners_3.endY, width=1.5,alpha=0.8,zorder=3,
                     headwidth=8, headlength=5, color='white', ax=ax['pitch'][0,1], label='Corners From Left')
        pitch.scatter(corners_3.next_shot_x, corners_3.next_shot_y,s=500,marker='p',color=team2_color,zorder=2,alpha=0.5, ax=ax['pitch'][0,1], edgecolors='white')

    if not corners_4.empty:
        pitch.arrows(corners_4.x, corners_4.y,
                     corners_4.endX, corners_4.endY, width=1.5,alpha=0.8,zorder=3,
                     headwidth=8, headlength=5, color='white', ax=ax['pitch'][1,1], label='Corners From Right')
        pitch.scatter(corners_4.next_shot_x, corners_4.next_shot_y,s=500,marker='p',color=team2_color,zorder=2,alpha=0.9, ax=ax['pitch'][1,1], edgecolors='white')

    pitch.scatter(80,45,color=team2_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][0,1])
    pitch.annotate(corner_summary_3['total_shots'][0],xy = (80,45),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,1])
    pitch.annotate('Shots',xy = (75,45),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,1])
    
    pitch.scatter(80,34,color=team2_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][0,1])
    pitch.annotate(corner_summary_3['shots_ot'][0],xy = (80,34),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,1])
    pitch.annotate('Shots(OT)',xy = (75,34),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,1])
    
    pitch.scatter(80,23,color=team2_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][0,1])
    pitch.annotate(corner_summary_3['goals'][0],xy = (80,23),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,1])
    pitch.annotate('Goals',xy = (75,23),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][0,1])
    
    
    pitch.scatter(80,45,color=team2_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][1,1])
    pitch.annotate(corner_summary_4['total_shots'][0],xy = (80,45),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,1])
    pitch.annotate('Shots',xy = (75,45),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,1])
    
    pitch.scatter(80,34,color=team2_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][1,1])
    pitch.annotate(corner_summary_4['shots_ot'][0],xy = (80,34),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,1])
    pitch.annotate('Shots(OT)',xy = (75,34),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,1])
    
    
    pitch.scatter(80,23,color=team2_color, s=1000, edgecolor='white', linewidth=2, alpha=0.7,marker='o',zorder=2,ax=ax['pitch'][1,1])
    pitch.annotate(corner_summary_4['goals'][0],xy = (80,23),color='white',ha='center', va='center',fontsize=15,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,1])
    pitch.annotate('Goals',xy = (75,23),color='white',ha='center', va='center',fontsize=12,zorder=3,fontproperties=font_prop,ax=ax['pitch'][1,1])

    #fig.savefig(f'C:/Users/acer/Documents/GitHub/IndianCitizen/ScorePredict/MatchReports/ACVInter/10.png',dpi=500,bbox_inches = 'tight',facecolor=background)

def plot_lost_pos(df,ax,team1_name,team2_name,team1_facecolor,team2_facecolor):
    df_lost_pos = df[
    ((df['type'] == 'Dispossessed') & (df['outcomeType'] == 'Successful')) |
    ((df['type'] == 'BallTouch') & (df['outcomeType'] == 'Unsuccessful'))
    ]
    
    # Group by 'teamName' and 'name', then get the counts
    grouped_df = df_lost_pos.groupby(['teamName', 'name']).size().reset_index(name='counts')
    
    # Filter for Team A
    team_a_df = grouped_df[grouped_df['teamName'] == team1_name]
    players_team_a = team_a_df['name'].tolist()
    counts_team_a = team_a_df['counts'].tolist()
    
    # Filter for Team B
    team_b_df = grouped_df[grouped_df['teamName'] == team2_name]
    players_team_b = team_b_df['name'].tolist()
    counts_team_b = team_b_df['counts'].tolist()
    # Comparing open play touches of both teams in each zones of the pitch, if more than 55% touches for a team it will be coloured of that team, otherwise gray to represent contested
    pcmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [team2_facecolor, 'gray', team1_facecolor], N=20)
    df1 = df[(df['teamName']==team1_name)]
    df2 = df[(df['teamName']==team2_name)]
    df2['x'] = 105-df2['x']
    df2['y'] =  68-df2['y']
    pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=background, line_color='white', linewidth=1, line_zorder=4)
    pitch.draw(ax=ax)
    #ax.set_ylim(-0.5,68.5)
    #ax.set_xlim(-0.5,105.5)

    pitch.scatter(df1.x, df1.y,s=500,marker='p',color=team1_facecolor,zorder=4,alpha=0.9, ax=ax, edgecolors='white')

    pitch.scatter(df2.x, df2.y,s=500,marker='p',color=team2_facecolor,zorder=4,alpha=0.9, ax=ax, edgecolors='white')
    ax.text(0,  -3, 'Attacking Direction--->', color=team1_facecolor, fontsize=15, ha='left', va='center',font_properties=font_prop)
    ax.text(105,-3, '<---Attacking Direction', color=team2_facecolor, fontsize=15, ha='right', va='center',font_properties=font_prop)

    ax.text(53,  73, 'Possession Lost', color='white', fontsize=30, ha='center', va='center',font_properties=font_prop)

    grouped_df = df.groupby(['teamName', 'name']).size().reset_index(name='counts')

    # Filter for Team A
    team_a_df = grouped_df[grouped_df['teamName'] == team1_name]
    players_team_a = team_a_df['name'].tolist()
    counts_team_a = team_a_df['counts'].tolist()
    
    # Filter for Team B
    team_b_df = grouped_df[grouped_df['teamName'] == team2_name]
    players_team_b = team_b_df['name'].tolist()
    counts_team_b = team_b_df['counts'].tolist()
    # Display the player names and counts for Team 1 on the left side
    for i, (player, count) in enumerate(zip(players_team_a, counts_team_a)):
        ax.text(-25, 68 - (i * 6.5), f"{player}: {count}", color=team1_facecolor, fontsize=15, ha='left', va='top', font_properties=font_prop)
    
    # Display the player names and counts for Team 2 on the right side
    for i, (player, count) in enumerate(zip(players_team_b, counts_team_b)):
        ax.text(130, 68 - (i * 6.5), f"{player}: {count}", color=team2_facecolor, fontsize=15, ha='right', va='top', font_properties=font_prop)
        
    return
