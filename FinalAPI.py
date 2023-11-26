import numpy as np
import pandas as pd
import pickle as pkl
import uvicorn
from fastapi import FastAPI

app = FastAPI()

toss_model = pkl.load(open('./pickles/toss_predict_model.pkl', 'rb'))
inning1_model = pkl.load(open('./pickles/inning1_run_predictor.pkl', 'rb'))
inning2_model = pkl.load(open('./pickles/inning2_run_predictor.pkl', 'rb'))
over_model = pkl.load(open('./pickles/over_predictor.pkl', 'rb'))
# Load the scaler object from the pickle file
with open('./pickles/run_wicket_scaler.pkl', 'rb') as scaler_file:
    run_wicket_scaler = pkl.load(scaler_file)
run_wicket_predictor = pkl.load(open('./pickles/run_wicket_predictor.pkl', 'rb'))

def team_mapping(team):
    team_dict =  {'Afghanistan': 0,
                'Australia': 1,
                'Bangladesh': 2,
                'England': 3,
                'India': 4,
                'Netherlands': 5,
                'New Zealand': 6,
                'Pakistan': 7,
                'South Africa': 8,
                'Sri Lanka': 9}
    team = team_dict[team]
    return team

def venue_mapping(venue):
    venue_dict = {'Arun Jaitley Stadium': 0,
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 1,
    'Eden Gardens': 2,
    'Himachal Pradesh Cricket Association Stadium': 3,
    'M Chinnaswamy Stadium': 4,
    'MA Chidambaram Stadium, Chepauk': 5,
    'Maharashtra Cricket Association Stadium': 6,
    'Narendra Modi Stadium': 7,
    'Rajiv Gandhi International Stadium': 8,
    'Wankhede Stadium': 9}
    venue = venue_dict[venue]
    return venue

def player_mapping(player_name):
    player_dict = {'A Dutt': 0,
    'A Zampa': 1,
    'AAP Atkinson': 2,
    'AD Mathews': 3,
    'AK Markram': 4,
    'AT Carey': 5,
    'AT Nidamanuru': 6,
    'AU Rashid': 7,
    'Abdullah Shafique': 8,
    'Azmatullah Omarzai': 9,
    'BA Stokes': 10,
    'BFW de Leede': 11,
    'BKG Mendis': 12,
    'Babar Azam': 13,
    'C Green': 14,
    'C Karunaratne': 15,
    'CAK Rajitha': 16,
    'CBRLS Kumara': 17,
    'CN Ackermann': 18,
    'CR Woakes': 19,
    'D Madushanka': 20,
    'DA Miller': 21,
    'DA Warner': 22,
    'DJ Malan': 23,
    'DJ Mitchell': 24,
    'DJ Willey': 25,
    'DM de Silva': 26,
    'DN Wellalage': 27,
    'DP Conway': 28,
    'FDM Karunaratne': 29,
    'Fakhar Zaman': 30,
    'Fazalhaq Farooqi': 31,
    'G Coetzee': 32,
    'GD Phillips': 33,
    'GJ Maxwell': 34,
    'H Klaasen': 35,
    'HC Brook': 36,
    'HE van der Dussen': 37,
    'HH Pandya': 38,
    'Haris Rauf': 39,
    'Hasan Ali': 40,
    'Hasan Mahmud': 41,
    'Hashmatullah Shahidi': 42,
    'Ibrahim Zadran': 43,
    'Iftikhar Ahmed': 44,
    'Ikram Alikhil': 45,
    'Imam-ul-Haq': 46,
    'Ishan Kishan': 47,
    'JC Buttler': 48,
    'JDS Neesham': 49,
    'JE Root': 50,
    'JJ Bumrah': 51,
    'JM Bairstow': 52,
    'JP Inglis': 53,
    'JR Hazlewood': 54,
    'K Rabada': 55,
    'KA Maharaj': 56,
    'KIC Asalanka': 57,
    'KL Rahul': 58,
    'KS Williamson': 59,
    'Kuldeep Yadav': 60,
    'L Ngidi': 61,
    'LH Ferguson': 62,
    'LS Livingstone': 63,
    'LV van Beek': 64,
    'M Jansen': 65,
    'M Labuschagne': 66,
    'M Pathirana': 67,
    'M Theekshana': 68,
    'MA Starc': 69,
    'MA Wood': 70,
    'MADI Hemantha': 71,
    'MD Shanaka': 72,
    'MDKJ Perera': 73,
    'MJ Henry': 74,
    'MJ Santner': 75,
    'MM Ali': 76,
    "MP O'Dowd": 77,
    'MP Stoinis': 78,
    'MR Marsh': 79,
    'MS Chapman': 80,
    'Mahedi Hasan': 81,
    'Mahmudullah': 82,
    'Mohammad Nabi': 83,
    'Mohammad Nawaz': 84,
    'Mohammad Rizwan': 85,
    'Mohammad Wasim': 86,
    'Mohammed Shami': 87,
    'Mujeeb Ur Rahman': 88,
    'Mushfiqur Rahim': 89,
    'Mustafizur Rahman': 90,
    'Najibullah Zadran': 91,
    'Nasum Ahmed': 92,
    'Naveen-ul-Haq': 93,
    'P Nissanka': 94,
    'PA van Meekeren': 95,
    'PJ Cummins': 96,
    'PVD Chameera': 97,
    'Q de Kock': 98,
    'R Klein': 99,
    'R Ravindra': 100,
    'RA Jadeja': 101,
    'RE van der Merwe': 102,
    'RG Sharma': 103,
    'RJW Topley': 104,
    'RR Hendricks': 105,
    'Rahmanullah Gurbaz': 106,
    'Rahmat Shah': 107,
    'Rashid Khan': 108,
    'S Samarawickrama': 109,
    'SA Edwards': 110,
    'SA Engelbrecht': 111,
    'SA Yadav': 112,
    'SM Curran': 113,
    'SPD Smith': 114,
    'SS Iyer': 115,
    'Saqib Zulfiqar': 116,
    'Saud Shakeel': 117,
    'Shadab Khan': 118,
    'Shaheen Shah Afridi': 119,
    'Shakib Al Hasan': 120,
    'Shariz Ahmad': 121,
    'Shoriful Islam': 122,
    'Shubman Gill': 123,
    'T Bavuma': 124,
    'T Shamsi': 125,
    'TA Boult': 126,
    'TG Southee': 127,
    'TM Head': 128,
    'TWM Latham': 129,
    'Tanzid Hasan': 130,
    'Taskin Ahmed': 131,
    'Towhid Hridoy': 132,
    'Usama Mir': 133,
    'V Kohli': 134,
    'Vikramjit Singh': 135,
    'W Barresi': 136,
    'WA Young': 137}
    player_name = player_dict[player_name]
    return player_name

def reverse_player_mapping(player_key):
    player_dict = {0: 'A Dutt',
    1: 'A Zampa',
    2: 'AAP Atkinson',
    3: 'AD Mathews',
    4: 'AK Markram',
    5: 'AT Carey',
    6: 'AT Nidamanuru',
    7: 'AU Rashid',
    8: 'Abdullah Shafique',
    9: 'Azmatullah Omarzai',
    10: 'BA Stokes',
    11: 'BFW de Leede',
    12: 'BKG Mendis',
    13: 'Babar Azam',
    14: 'C Green',
    15: 'C Karunaratne',
    16: 'CAK Rajitha',
    17: 'CBRLS Kumara',
    18: 'CN Ackermann',
    19: 'CR Woakes',
    20: 'D Madushanka',
    21: 'DA Miller',
    22: 'DA Warner',
    23: 'DJ Malan',
    24: 'DJ Mitchell',
    25: 'DJ Willey',
    26: 'DM de Silva',
    27: 'DN Wellalage',
    28: 'DP Conway',
    29: 'FDM Karunaratne',
    30: 'Fakhar Zaman',
    31: 'Fazalhaq Farooqi',
    32: 'G Coetzee',
    33: 'GD Phillips',
    34: 'GJ Maxwell',
    35: 'H Klaasen',
    36: 'HC Brook',
    37: 'HE van der Dussen',
    38: 'HH Pandya',
    39: 'Haris Rauf',
    40: 'Hasan Ali',
    41: 'Hasan Mahmud',
    42: 'Hashmatullah Shahidi',
    43: 'Ibrahim Zadran',
    44: 'Iftikhar Ahmed',
    45: 'Ikram Alikhil',
    46: 'Imam-ul-Haq',
    47: 'Ishan Kishan',
    48: 'JC Buttler',
    49: 'JDS Neesham',
    50: 'JE Root',
    51: 'JJ Bumrah',
    52: 'JM Bairstow',
    53: 'JP Inglis',
    54: 'JR Hazlewood',
    55: 'K Rabada',
    56: 'KA Maharaj',
    57: 'KIC Asalanka',
    58: 'KL Rahul',
    59: 'KS Williamson',
    60: 'Kuldeep Yadav',
    61: 'L Ngidi',
    62: 'LH Ferguson',
    63: 'LS Livingstone',
    64: 'LV van Beek',
    65: 'M Jansen',
    66: 'M Labuschagne',
    67: 'M Pathirana',
    68: 'M Theekshana',
    69: 'MA Starc',
    70: 'MA Wood',
    71: 'MADI Hemantha',
    72: 'MD Shanaka',
    73: 'MDKJ Perera',
    74: 'MJ Henry',
    75: 'MJ Santner',
    76: 'MM Ali',
    77: "MP O'Dowd",
    78: 'MP Stoinis',
    79: 'MR Marsh',
    80: 'MS Chapman',
    81: 'Mahedi Hasan',
    82: 'Mahmudullah',
    83: 'Mohammad Nabi',
    84: 'Mohammad Nawaz',
    85: 'Mohammad Rizwan',
    86: 'Mohammad Wasim',
    87: 'Mohammed Shami',
    88: 'Mujeeb Ur Rahman',
    89: 'Mushfiqur Rahim',
    90: 'Mustafizur Rahman',
    91: 'Najibullah Zadran',
    92: 'Nasum Ahmed',
    93: 'Naveen-ul-Haq',
    94: 'P Nissanka',
    95: 'PA van Meekeren',
    96: 'PJ Cummins',
    97: 'PVD Chameera',
    98: 'Q de Kock',
    99: 'R Klein',
    100: 'R Ravindra',
    101: 'RA Jadeja',
    102: 'RE van der Merwe',
    103: 'RG Sharma',
    104: 'RJW Topley',
    105: 'RR Hendricks',
    106: 'Rahmanullah Gurbaz',
    107: 'Rahmat Shah',
    108: 'Rashid Khan',
    109: 'S Samarawickrama',
    110: 'SA Edwards',
    111: 'SA Engelbrecht',
    112: 'SA Yadav',
    113: 'SM Curran',
    114: 'SPD Smith',
    115: 'SS Iyer',
    116: 'Saqib Zulfiqar',
    117: 'Saud Shakeel',
    118: 'Shadab Khan',
    119: 'Shaheen Shah Afridi',
    120: 'Shakib Al Hasan',
    121: 'Shariz Ahmad',
    122: 'Shoriful Islam',
    123: 'Shubman Gill',
    124: 'T Bavuma',
    125: 'T Shamsi',
    126: 'TA Boult',
    127: 'TG Southee',
    128: 'TM Head',
    129: 'TWM Latham',
    130: 'Tanzid Hasan',
    131: 'Taskin Ahmed',
    132: 'Towhid Hridoy',
    133: 'Usama Mir',
    134: 'V Kohli',
    135: 'Vikramjit Singh',
    136: 'W Barresi',
    137: 'WA Young'}
    player_name = player_dict[player_key]
    return player_name


def reverse_team_mapping(team):
    batting_team_dict = {0: 'Afghanistan',
                         1: 'Australia',
                         2: 'Bangladesh',
                         3: 'England',
                         4: 'India',
                         5: 'Netherlands',
                         6: 'New Zealand',
                         7: 'Pakistan',
                         8: 'South Africa',
                         9: 'Sri Lanka'}
    team_name=batting_team_dict[team]
    return team_name

def update_table(winner, loser, winner_runs, winner_overs, loser_runs, loser_overs, points_df):
    winner_runs = winner_runs[0][0]
    winner_overs = winner_overs[0][0]
    loser_runs = loser_runs[0][0]
    loser_overs = loser_overs[0][0]
    # get the winner and loser index from points_df dataframe
    winner_index = points_df[points_df['Team'] == winner].index[0]
    loser_index = points_df[points_df['Team'] == loser].index[0]

    points_df.loc[winner_index, 'Series Form'] = "W"+points_df.loc[winner_index, 'Series Form']
    points_df.loc[loser_index, 'Series Form'] = "L"+points_df.loc[loser_index, 'Series Form']

    points_df.loc[winner_index, 'Points'] += 2
    points_df.loc[winner_index, 'Matches'] += 1
    points_df.loc[winner_index, 'Won'] += 1

    points_df.loc[loser_index, 'Matches'] += 1
    points_df.loc[loser_index, 'Lost'] += 1

    for_runs = points_df.loc[winner_index, 'For_Runs']
    for_overs = points_df.loc[winner_index, 'For_Overs']

    for_runs += winner_runs
    for_overs += winner_overs

    against_runs = points_df.loc[winner_index, 'Against_Runs']
    against_overs = points_df.loc[winner_index, 'Against_Overs']

    against_runs += loser_runs
    against_overs += loser_overs

    print(winner_runs, winner_overs, loser_runs, loser_overs)
    print(for_runs, for_overs, against_runs, against_overs)
    print(points_df.loc[winner_index, 'For_Runs'])

    points_df.loc[winner_index, 'For_Runs'] = for_runs
    points_df.loc[winner_index, 'For_Overs'] = for_overs
    points_df.loc[winner_index, 'Against_Runs'] = against_runs
    points_df.loc[winner_index, 'Against_Overs'] = against_overs
    points_df.loc[winner_index, 'Net Run Rate'] = (for_runs/for_overs) - (against_runs/against_overs)

    for_runs = points_df.loc[loser_index, 'For_Runs']
    for_overs = points_df.loc[loser_index, 'For_Overs']

    for_runs += loser_runs
    for_overs += loser_overs

    against_runs = points_df.loc[loser_index, 'Against_Runs']
    against_overs = points_df.loc[loser_index, 'Against_Overs']

    against_runs += winner_runs
    against_overs += winner_overs

    points_df.loc[loser_index, 'For_Runs'] = for_runs
    points_df.loc[loser_index, 'For_Overs'] = for_overs
    points_df.loc[loser_index, 'Against_Runs'] = against_runs
    points_df.loc[loser_index, 'Against_Overs'] = against_overs
    points_df.loc[loser_index, 'Net Run Rate'] = (for_runs/for_overs) - (against_runs/against_overs)
    return points_df


def complete_match(team1, team2, venue):
        num = np.random.uniform(0, 1)
        team1 = team_mapping(team1)
        team2 = team_mapping(team2)
        venue = venue_mapping(venue)
        toss_winner, toss_losser = 0, 0
        if num < 0.5:
            toss_winner = team1
            toss_losser = team2

        toss_df = pd.DataFrame([[team1, team2, venue, toss_winner]], columns=['team1', 'team2', 'venue', 'toss_winner'])
        # predict the toss_winners_decision
        toss_winners_decision = toss_model.predict(toss_df)
        toss_winners_decision = np.round(toss_winners_decision)

        if toss_winners_decision == 0:
            batting_team = toss_winner
            bowling_team = toss_losser
        else:
            batting_team = toss_losser
            bowling_team = toss_winner

        inning1_df = pd.DataFrame([[venue, batting_team, bowling_team, 50]], columns=['venue', 'batting_team', 'bowling_team', 'total_overs_played'])
        inning1_score = inning1_model.predict(inning1_df.astype('float32'))

        inning1_df = pd.DataFrame([[1, venue, batting_team, bowling_team, inning1_score]], columns=['innings', 'venue', 'batting_team', 'bowling_team', 'total_runs_per_innings_match'])
        inning1_over = over_model.predict(inning1_df.astype('float32'))

        if inning1_over > 50:
            inning1_score = inning1_score*50/inning1_over
            inning1_over = 50
        
        inning2_df = pd.DataFrame([[venue, bowling_team, batting_team, 50, inning1_score]], columns=['venue', 'batting_team', 'bowling_team', 'total_overs_played', 'total_runs_in_innings1'])
        inning2_score = inning2_model.predict(inning2_df.astype('float32'))

        inning2_df = pd.DataFrame([[2, venue, bowling_team, batting_team, inning2_score]], columns=['innings', 'venue', 'batting_team', 'bowling_team', 'total_runs_per_innings_match'])
        inning2_over = over_model.predict(inning2_df.astype('float32'))

        if inning2_over > 50:
            inning2_score = inning2_score*50/inning2_over
            inning2_over = 50
    
        return inning1_score, inning1_over, inning2_score, inning2_over, batting_team, bowling_team


def predict_semifinalists():
    upcoming_matches = pd.read_csv('./csvs/upcoming_matches.csv')

    points_df = pd.read_csv('./csvs/points_table.csv')

    points_df[['For_Runs', 'For_Overs']] = points_df['For'].str.split('/', expand=True)
    points_df['For_Runs'] = points_df['For_Runs'].astype(int)
    points_df['For_Overs'] = points_df['For_Overs'].astype(float)
    points_df.drop(['For'], axis=1, inplace=True)

    points_df[['Against_Runs', 'Against_Overs']] = points_df['Against'].str.split('/', expand=True)
    points_df['Against_Runs'] = points_df['Against_Runs'].astype(int)
    points_df['Against_Overs'] = points_df['Against_Overs'].astype(float)
    points_df.drop(['Against'], axis=1, inplace=True)

    for team1, team2, venue in zip(upcoming_matches['team1'], upcoming_matches['team2'], upcoming_matches['venue']):
        inning1_score, inning1_over, inning2_score, inning2_over, batting_team, bowling_team = complete_match(team1, team2, venue)
        if inning1_score > inning2_score:
            winner = reverse_team_mapping(batting_team)
            points_df = update_table(winner, reverse_team_mapping(bowling_team), inning1_score, inning1_over, inning2_score, inning2_over, points_df)
        else:
            winner = reverse_team_mapping(bowling_team)
            points_df = update_table(winner, reverse_team_mapping(batting_team), inning2_score, inning2_over, inning1_score, inning1_over, points_df)
    
    points_df = points_df.sort_values(by=['Points', 'Net Run Rate'], ascending=False)

    # # create a csv file with top 4 teams
    points_df.head(4).to_csv('./csvs/semifinalists.csv', index=False)

    return points_df.iloc[:4]['Team'].values.tolist()

def predict_finalists():
    current_table = pd.read_csv('./csvs/semifinalists.csv')

    # Semifinal-1
    team1 = current_table.iloc[0]['Team']
    team2 = current_table.iloc[3]['Team']
    venue = 'Wankhede Stadium'
    inning1_score, inning1_over, inning2_score, inning2_over, batting_team, bowling_team = complete_match(team1, team2, venue)
    if inning1_score > inning2_score:
        semifinal1_winner = reverse_team_mapping(batting_team)
    else:
        semifinal1_winner = reverse_team_mapping(bowling_team)
    
    # Semifinal-2
    team1 = current_table.iloc[1]['Team']
    team2 = current_table.iloc[2]['Team']
    venue = 'Eden Gardens'
    inning1_score, inning1_over, inning2_score, inning2_over, batting_team, bowling_team = complete_match(team1, team2, venue)
    if inning1_score > inning2_score:
        semifinal2_winner = reverse_team_mapping(batting_team)
    else:
        semifinal2_winner = reverse_team_mapping(bowling_team)

    pd.DataFrame({'Team': [semifinal1_winner, semifinal2_winner]}).to_csv('./csvs/finalists.csv', index=False)
    
    finalists_team = []
    # add both teams to finalists_team
    finalists_team.append(semifinal1_winner)
    finalists_team.append(semifinal2_winner)

    return finalists_team

def predict_winner():
    current_table = pd.read_csv('./csvs/finalists.csv')

    # Final
    team1 = current_table.iloc[0]['Team']
    team2 = current_table.iloc[1]['Team']
    venue = 'Narendra Modi Stadium'
    inning1_score, inning1_over, inning2_score, inning2_over, batting_team, bowling_team = complete_match(team1, team2, venue)
    if inning1_score > inning2_score:
        winner = reverse_team_mapping(batting_team)
    else:
        winner = reverse_team_mapping(bowling_team)
    
    pd.DataFrame({'Team': [winner]}).to_csv('./csvs/winner.csv', index=False)

    return winner

def predict_eleven():
    # load the player_details.csv
    player_details = pd.read_csv('./csvs/player_details.csv')
    
    finalists = pd.read_csv('./csvs/finalists.csv')

    # change the column name from 'team' to 'Team'
    player_details.rename(columns={'team': 'Team'}, inplace=True)

    finalists = finalists['Team'].values.tolist()
    
    team1_players = player_details[player_details['Team'] == finalists[0]]
    team2_players = player_details[player_details['Team'] == finalists[1]]
    
    # keep only those rows in team1 where opponent_team is finalists[1]
    team1_players = team1_players[team1_players['opponent_team'] == finalists[1]]
    # keep only those rows in team2 where opponent_team is finalists[0]
    team2_players = team2_players[team2_players['opponent_team'] == finalists[0]]

    # print the type of run_wicket_scaler
    print(type(run_wicket_scaler))
    
    team1_players[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = run_wicket_scaler.transform(team1_players[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])
    team2_players[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = run_wicket_scaler.transform(team2_players[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])
    
    # remove match_runs and match_wickets from the both dataframe
    team1_players = team1_players.drop(['match_runs', 'match_wickets'], axis=1)
    team2_players = team2_players.drop(['match_runs', 'match_wickets'], axis=1)
    
    # encode player_name for both the dataframe
    team1_players['player_name'] = team1_players['player_name'].apply(player_mapping)
    team2_players['player_name'] = team2_players['player_name'].apply(player_mapping)
    
    # encode team for both the dataframe
    team1_players['Team'] = team1_players['Team'].apply(team_mapping)
    team2_players['Team'] = team2_players['Team'].apply(team_mapping)
    team1_players['opponent_team'] = team1_players['opponent_team'].apply(team_mapping)
    team2_players['opponent_team'] = team2_players['opponent_team'].apply(team_mapping)
    
    # encode venue for both the dataframe
    team1_players['venue'] = team1_players['venue'].apply(venue_mapping)
    team2_players['venue'] = team2_players['venue'].apply(venue_mapping)
    
    # make a prediction for team1_players using run_wicket_predictor which will give two columns match_runs','match_wickets'
    team1_players[['match_runs','match_wickets']] = run_wicket_predictor.predict(team1_players)
    # make a prediction for team2_players using run_wicket_predictor which will give two columns match_runs','match_wickets'
    team2_players[['match_runs','match_wickets']] = run_wicket_predictor.predict(team2_players)
    
    team1_final_eleven = []
    team2_final_eleven = []
    
    # pick top 6 batsman from team1_players and add it to team1_final_eleven
    team1_final_eleven.extend(team1_players.sort_values(by=['match_runs'], ascending=False).iloc[:6]['player_name'].values.tolist())
    # pick top 5 bowlers from team1_players and add it to team1_final_eleven
    team1_final_eleven.extend(team1_players.sort_values(by=['match_wickets'], ascending=False).iloc[:5]['player_name'].values.tolist())
    
    # pick top 6 batsman from team2_players and add it to team2_final_eleven
    team2_final_eleven.extend(team2_players.sort_values(by=['match_runs'], ascending=False).iloc[:6]['player_name'].values.tolist())
    # pick top 5 bowlers from team2_players and add it to team2_final_eleven
    team2_final_eleven.extend(team2_players.sort_values(by=['match_wickets'], ascending=False).iloc[:5]['player_name'].values.tolist())
    
    # make reverse mapping of player for both final eleven
    team1_final_eleven = [reverse_player_mapping(player) for player in team1_final_eleven]
    team2_final_eleven = [reverse_player_mapping(player) for player in team2_final_eleven]
    
    # make a dataframe with team1_final_eleven and team2_final_eleven
    final_eleven = pd.DataFrame({'team1': [team1_final_eleven], 'team2': [team2_final_eleven]})
    
    return final_eleven

def top_batsman():
    player_df = pd.read_csv('./csvs/player_details.csv')
    top_run_scorer = player_df.groupby(['player_name'])['match_runs'].sum().reset_index()
    top_run_scorer = top_run_scorer.sort_values(by='match_runs', ascending=False)
    top_run_scorer = top_run_scorer.head(15)
    # print(top_run_scorer)
    player_df.drop(['total_runs','match_runs'], axis=1, inplace=True)
    player_df = player_df[player_df['player_name'].isin(top_run_scorer['player_name'])]
    # print(player_df)
    player_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    # print(top_run_scorer.columns)
    player_df = player_df.merge(top_run_scorer, on='player_name', how='left')

    player_df.rename(columns={'match_runs': 'total_runs'}, inplace=True)
    player_df = player_df.sort_values(by='total_runs', ascending=False)
    player_df.drop(['opponent_team', 'venue'], axis=1, inplace=True)

    upcoming_matches = pd.read_csv('./csvs/upcoming_matches.csv')

    predict_df = pd.DataFrame(columns=['player_name','Team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'])

    for index, player_row in player_df.iterrows():
        for index, match_row in upcoming_matches.iterrows():
            if player_row['Team'] == match_row['team1'] or player_row['Team'] == match_row['team2']:
                opponent_team = match_row['team1'] if player_row['Team'] == match_row['team2'] else match_row['team2']
                predict_df = predict_df._append({'player_name': player_row['player_name'], 'Team': player_row['Team'], 'opponent_team':opponent_team, 'venue': match_row['venue'], 'total_runs': player_row['total_runs'], 'highest_score': player_row['highest_score'], 'batting_avg': player_row['batting_avg'], 'strike_rate': player_row['strike_rate'], 'bowling_runs': player_row['bowling_runs'], 'total_wickets': player_row['total_wickets'], 'bowling_avg': player_row['bowling_avg'], 'economy': player_row['economy']}, ignore_index=True)

    predict_df = predict_df.reset_index(drop=True)
    final_df = predict_df.copy()
    
    # encode player_name for both the dataframe
    predict_df['player_name'] = predict_df['player_name'].apply(player_mapping)
    # encode team for the dataframe
    predict_df['Team'] = predict_df['Team'].apply(team_mapping)
    # encode opponent_tema for the dataframe
    predict_df['opponent_team'] = predict_df['opponent_team'].apply(team_mapping)
    # encode venue for the dataframe
    predict_df['venue'] = predict_df['venue'].apply(venue_mapping)

    predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = run_wicket_scaler.transform(predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])
    predict_df[['player_name','Team','opponent_team','venue']] = predict_df[['player_name','Team','opponent_team','venue']].astype(int)
    predict_df = predict_df[['player_name','Team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy']]
    runs = run_wicket_predictor.predict(predict_df)
    runs = runs.astype(int)
    runs = runs[:,0]
    final_df['runs'] = runs
    final_df['predicted_runs'] = final_df.groupby('player_name')['runs'].transform('sum')
    final_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    final_df['total_runs'] = final_df['total_runs'] + final_df['predicted_runs']
    final_df.drop(['predicted_runs','runs','Team','opponent_team','venue','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'], axis=1, inplace=True)
    final_df['answer'] = final_df['player_name'] + ' - ' + final_df['total_runs'].astype(str)
    return final_df['answer']

def top_bowler():
    player_df = pd.read_csv('./csvs/player_details.csv')

    top_wicket_scorer = player_df.groupby(['player_name'])['match_wickets'].sum().reset_index()
    top_wicket_scorer = top_wicket_scorer.sort_values(by='match_wickets', ascending=False)
    top_wicket_scorer = top_wicket_scorer.head(15)
    player_df.drop(['total_wickets','match_wickets'], axis=1, inplace=True)
    player_df = player_df[player_df['player_name'].isin(top_wicket_scorer['player_name'])]
    player_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    player_df = player_df.merge(top_wicket_scorer, on='player_name', how='left')

    player_df.rename(columns={'match_wickets': 'total_wickets'}, inplace=True)
    player_df = player_df.sort_values(by='total_wickets', ascending=False)
    player_df.drop(['opponent_team', 'venue'], axis=1, inplace=True)

    upcoming_matches = pd.read_csv('./csvs/upcoming_matches.csv')

    predict_df = pd.DataFrame(columns=['player_name','Team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy'])

    for index, player_row in player_df.iterrows():
        for index, match_row in upcoming_matches.iterrows():
            if player_row['Team'] == match_row['team1'] or player_row['Team'] == match_row['team2']:
                opponent_team = match_row['team1'] if player_row['Team'] == match_row['team2'] else match_row['team2']
                predict_df = predict_df._append({'player_name': player_row['player_name'], 'Team': player_row['Team'], 'opponent_team':opponent_team, 'venue': match_row['venue'], 'total_runs': player_row['total_runs'], 'highest_score': player_row['highest_score'], 'batting_avg': player_row['batting_avg'], 'strike_rate': player_row['strike_rate'], 'bowling_runs': player_row['bowling_runs'], 'total_wickets': player_row['total_wickets'], 'bowling_avg': player_row['bowling_avg'], 'economy': player_row['economy']}, ignore_index=True)

    predict_df = predict_df.reset_index(drop=True)
    final_df = predict_df.copy()
    # encode player_name for both the dataframe
    predict_df['player_name'] = predict_df['player_name'].apply(player_mapping)
    # encode team for the dataframe
    predict_df['Team'] = predict_df['Team'].apply(team_mapping)
    # encode opponent_tema for the dataframe
    predict_df['opponent_team'] = predict_df['opponent_team'].apply(team_mapping)
    # encode venue for the dataframe
    predict_df['venue'] = predict_df['venue'].apply(venue_mapping)

    predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = run_wicket_scaler.transform(predict_df[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])
    predict_df[['player_name','Team','opponent_team','venue']] = predict_df[['player_name','Team','opponent_team','venue']].astype(int)
    predict_df = predict_df[['player_name','Team','opponent_team','venue','total_runs','highest_score','batting_avg','strike_rate','bowling_runs','total_wickets','bowling_avg','economy']]
    wickets = run_wicket_predictor.predict(predict_df)
    wickets = wickets.astype(int)
    wickets = wickets[:,1]
    wickets = np.where(wickets < 0, 0, wickets)
    final_df['wickets'] = wickets
    final_df['predicted_wickets'] = final_df.groupby('player_name')['wickets'].transform('sum')
    final_df.drop_duplicates(subset=['player_name'], keep='first', inplace=True)
    final_df['total_wickets'] = final_df['total_wickets'] + final_df['predicted_wickets']
    final_df.drop(['predicted_wickets','wickets','Team','opponent_team','venue','highest_score','batting_avg','strike_rate','bowling_runs','bowling_avg','economy','total_runs'], axis=1, inplace=True)
    final_df['answer'] = final_df['player_name'] + ' - ' + final_df['total_wickets'].astype(str)
    return final_df['answer']

@app.get("/predict-winner")
def predict_winner_api_function():
    winner = predict_winner()
    print(winner)
    return {"winner": winner}

@app.get("/predict-semifinalists")
def predict_semifinalists_api_function():
    return {"semi-finalist": predict_semifinalists()}

@app.get("/predict-finalists")
def predict_finalists_api_function():
    return {"finalist": predict_finalists()}

@app.get("/predict-eleven")
def predict_eleven_api_function():
    eleven_df = predict_eleven()
    team1 = eleven_df['team1'].tolist()
    team2 = eleven_df['team2'].tolist()
    return {"team1": team1, "team2": team2}

@app.get("/predict-batsman")
def predict_batsman_api_function():
    answer = top_batsman().tolist()
    return {"Top Scorers": answer}

@app.get("/predict-bowler")
def predict_batsman_api_function():
    answer = top_bowler().tolist()
    return {"Top Wicket Tackers": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)