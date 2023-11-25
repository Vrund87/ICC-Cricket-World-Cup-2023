from fastapi import FastAPI
import uvicorn
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shutil
import joblib

toss_scaler = joblib.load('./pickles/toss_scaler.pkl')
inning_1_scaler = joblib.load('./pickles/scaler_inning1.pkl')
inning_2_scaler = joblib.load('./pickles/scaler_inning2.pkl')
player_scaler = joblib.load('./pickles/player_scaler.pkl')

app=FastAPI()
overs_model = pkl.load(open('./pickles/over_model.pkl', 'rb'))
inning1_model = pkl.load(open('./pickles/inning_1_run.pkl', 'rb'))
inning2_model = pkl.load(open('./pickles/inning_2_run.pkl', 'rb'))
toss_model = pkl.load(open('./pickles/toss_prediction.pkl', 'rb'))
player_model = pkl.load(open('./pickles/player_model.pkl', 'rb'))

app =FastAPI()

def get_percentages(team_array, opposition_array):
    # Read the CSV data
    df = pd.read_csv('./csvs/previous_data_per.csv')

    # Initialize an array to store the percentages
    result_array = []

    # Iterate through the input arrays
    for team, opposition in zip(team_array, opposition_array):
        # Search for the corresponding row in the DataFrame
        match_row = df[(df['Team'] == team) & (df['Opposition'] == opposition)]

        # If a match is found, add the actual % value; otherwise, add 0.5
        if not match_row.empty:
            value = match_row['%'].values[0]
            if value == '-':
                result_array.append(0.5)
            else:
                result_array.append(float(value)/100)
        else:
            result_array.append(0.5)

    return result_array

def get_semi_percentages(team_array, opposition_array):
    # Read the CSV data
    df = pd.read_csv('./csvs/previous_data_per.csv')
    recent = pd.read_csv('./csvs/matches.csv')

    # Initialize an array to store the percentages
    result_array = []
    # Iterate through the input arrays
    for team, opposition in zip(team_array, opposition_array):
        match_row = df[(df['Team'] == reverse_mappings(team)) & (df['Opposition'] == reverse_mappings(opposition))]

        winnr = recent[(recent['team1'] == reverse_mappings(team)) & (recent['team2'] == reverse_mappings(opposition))]

        if winnr.empty:
            winnr = recent[(recent['team1'] == reverse_mappings(opposition)) & (recent['team2'] == reverse_mappings(team))]

        # If a match is found, add the actual % value; otherwise, add 0.5
        if not match_row.empty:
            value = match_row['%'].values[0]
            if value == '-':
                result_array.append(0.5)
            else:
                if not winnr.empty:
                    winnr = winnr['winner'].values[0]
                    if winnr == reverse_mappings(team):
                        result_array.append((float(value)/100)+0.15)
                    else:
                        result_array.append((float(value)/100)-0.15)
                else:
                    result_array.append((float(value)/100))
        else:
            result_array.append(0.5)
    return result_array

def get_final_percentages(team, opposition):
    # Read the CSV data
    df = pd.read_csv('./csvs/previous_data_per.csv')
    recent = pd.read_csv('./csvs/matches.csv')

    # print(team_array, opposition_array)

    # Initialize an array to store the percentages
    result_array = []

    # Search for the corresponding row in the DataFrame

    match_row = df[(df['Team'] == reverse_mappings(team)) & (df['Opposition'] == reverse_mappings(opposition))]
    winnr = recent[(recent['team1'] == reverse_mappings(team)) & (recent['team2'] == reverse_mappings(opposition))]
    # winnr = winnr['winner'].values

    if match_row.empty:
        match_row = df[(df['Team'] == reverse_mappings(opposition)) & (df['Opposition'] == reverse_mappings(team))]

    if winnr.empty:
        winnr = recent[(recent['team1'] == reverse_mappings(opposition)) & (recent['team2'] == reverse_mappings(team))]

 
    # If a match is found, add the actual % value; otherwise, add 0.5
    if not match_row.empty:
        value = match_row['%'].values[0]
        if value == '-':
            result_array.append(0.5)
        else:
            if winnr.empty:
                result_array.append((float(value)/100))
            else:
                winnr = winnr['winner'].values[0]
                if winnr == reverse_mappings(team):
                    result_array.append((float(value)/100 + 0.15))
                else:
                    result_array.append((float(value)/100 - 0.15))
    else:
        result_array.append(0.5)

    return result_array

# To predict the resultant score of the first innings without over constraints
def predict_inning1(df_inning1,toss_mapping,number_of_matches):
    # print("In function Predict inning 1 score\n")
    feature=list(df_inning1.columns)
    number_of_features=df_inning1.shape[1]
    for i in range(0,number_of_features-2):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df_inning1.loc[j, feature_name]
            if feature_name in toss_mapping and original_value in toss_mapping[feature_name]:
                new_value = toss_mapping[feature_name][original_value]
                df_inning1.loc[j, feature_name] = new_value

    df_inning1.rename(columns={"team1":"batting_team","team2":"bowling_team","venue_x":"venue","%":"%"},inplace=True)
    df_inning1['first_inning_score'] = df_inning1['first_inning_score'].astype(float)
    #convert batting_team and bowling_team to int
    df_inning1['batting_team'] = df_inning1['batting_team'].astype(int)
    df_inning1['bowling_team'] = df_inning1['bowling_team'].astype(int)
    df_inning1['venue'] = df_inning1['venue'].astype(int)
    df_inning1[['total_overs_played','first_inning_score']] = inning_1_scaler.transform(df_inning1[['total_overs_played','first_inning_score']])
    inning1_pred=inning1_model.predict(df_inning1.astype(float))
    inning1_pred = np.round(inning1_pred)
    return inning1_pred

#to predict the overs required to score the predicted score
def over_predict_inning1(df_inning1,toss_mapping,number_of_matches,inning1_pred):
    df_inning1['total_runs_per_innings_match'] = list(inning1_pred)
    df_inning1.drop('total_overs_played', axis='columns', inplace=True)
    df_inning1['innings'] = [1 for i in range(0, number_of_matches)]
    # Reorder the columns
    df_inning1 = df_inning1.reindex(columns=['innings', 'venue', 'batting_team', 'bowling_team', 'total_runs_per_innings_match', '%'])

    # print(df_inning1.astype(float))

    overs_pred1 = overs_model.predict(df_inning1.astype(float))

    for i in range(0, len(inning1_pred)):
        if overs_pred1[i] > 50.0:
            inning1_pred[i] = inning1_pred[i] * 50 / overs_pred1[i]
            overs_pred1[i] = 50.0

    return inning1_pred, overs_pred1

def predict_inning2(df_inning2,toss_mapping,number_of_matches,inning1_pred):
    feature=list(df_inning2.columns)
    number_of_features=df_inning2.shape[1]
    for i in range(0,number_of_features-2):
        for j in range(0,number_of_matches):
            feature_name = feature[i]
            original_value = df_inning2.loc[j, feature_name]
            if feature_name in toss_mapping and original_value in toss_mapping[feature_name]:
                new_value = toss_mapping[feature_name][original_value]
                df_inning2.loc[j, feature_name] = new_value

    df_inning2['total_runs_in_innings1'] = list(inning1_pred)
    #round off to nearest integer
    df_inning2['total_runs_in_innings1'] = df_inning2['total_runs_in_innings1'].astype(int)
    df_inning2.rename(columns={"team1":"batting_team","team2":"bowling_team","venue_x":"venue","total_runs_in_innings1":"total_runs_in_innings1","%":"%"},inplace=True)
    df_inning2 = df_inning2.reindex(columns=['venue', 'batting_team', 'bowling_team', 'total_overs_played','first_inning_score','second_inning_score','total_runs_in_innings1', '%'])
    df_inning2[['total_overs_played','first_inning_score','second_inning_score','total_runs_in_innings1']] = inning_2_scaler.transform(df_inning2[['total_overs_played','first_inning_score','second_inning_score','total_runs_in_innings1']])

    #save this df_inning2 to csv
    # df_inning2.to_csv('./csvs/dummy_2.csv', index=False)
    # print(df_inning2.dtypes)
    df_inning2['batting_team'] = df_inning2['batting_team'].astype(int)
    df_inning2['bowling_team'] = df_inning2['bowling_team'].astype(int)
    inning2_pred=inning2_model.predict(df_inning2.astype(float))
    #round off to nearest integer
    inning2_pred = np.round(inning2_pred)
    return inning2_pred

def update_points(points_df, winner, loser, winner_runs, winner_overs, loser_runs, loser_overs):
    # Read the CSV data

    # print(winner, loser)
    # Read the CSV data
    df = pd.read_csv('./csvs/prev_data_per.csv')
    # Update the % column for winner vs loser
    if not df.loc[(df['Team'] == winner) & (df['Opposition'] == loser), '%'].empty:
       df.loc[(df['Team'] == winner) & (df['Opposition'] == loser), '%'] = float(df.loc[(df['Team'] == winner) & (df['Opposition'] == loser), '%'].values[0]) + 15

    # Update the % column for loser vs winner
    if not df.loc[(df['Team'] == loser) & (df['Opposition'] == winner), '%'].empty:
        df.loc[(df['Team'] == loser) & (df['Opposition'] == winner), '%'] = float(df.loc[(df['Team'] == winner) & (df['Opposition'] == loser), '%'].values[0]) - 15

    # Write the updated data back to the CSV file
    df.to_csv('./csvs/../prev_data_per.csv', index=False)

    # Get the indices of the winner and loser
    winner_index = points_df[points_df['Team'] == winner].index.values[0]
    loser_index = points_df[points_df['Team'] == loser].index.values[0]

    points_df.loc[winner_index, 'Series Form'] = points_df.loc[winner_index, 'Series Form']+"W"
    points_df.loc[loser_index, 'Series Form'] = points_df.loc[loser_index, 'Series Form']+"L"

    # Update the winner's points
    points_df.loc[winner_index, 'Points'] += 2
    points_df.loc[winner_index, 'Matches'] += 1
    points_df.loc[winner_index, 'Won'] += 1

    # Update the loser's points
    points_df.loc[loser_index, 'Matches'] += 1
    points_df.loc[loser_index, 'Lost'] += 1

    # update the NRR of winner and loser by adding score/run upon cum overs - opponents score/run upon cum overs
    # Update the winner's points
    split_winner_for = points_df.loc[winner_index, 'For']
    for_runs = int(float(split_winner_for.split('/')[0]))
    for_overs = float(split_winner_for.split('/')[1])

    for_runs += winner_runs
    for_overs += winner_overs

    split_winner_against = points_df.loc[winner_index, 'Against']
    against_runs = int(float(split_winner_against.split('/')[0]))
    against_overs = float(split_winner_against.split('/')[1])

    against_runs += loser_runs
    against_overs += loser_overs

    points_df.loc[winner_index, 'For'] = str(for_runs) + '/' + str(for_overs)
    points_df.loc[winner_index, 'Against'] = str(against_runs) + '/' + str(against_overs)    

    points_df.loc[winner_index, 'Net Run Rate'] = (for_runs/for_overs) - (against_runs/against_overs)

    # Update the loser's points
    split_loser_for = points_df.loc[loser_index, 'For']
    for_runs = int(float(split_winner_for.split('/')[0]))
    for_overs = float(split_winner_for.split('/')[1])

    for_runs += loser_runs
    for_overs += loser_overs

    split_loser_against = points_df.loc[winner_index, 'Against']
    against_runs = int(float(split_loser_for.split('/')[0]))
    against_overs = float(split_loser_against.split('/')[1])

    against_runs += winner_runs
    against_overs += winner_overs

    points_df.loc[winner_index, 'For'] = str(for_runs) + '/' + str(for_overs)
    points_df.loc[winner_index, 'Against'] = str(against_runs) + '/' + str(against_overs)    

    points_df.loc[loser_index, 'Net Run Rate'] = (for_runs/for_overs) - (against_runs/against_overs)

def final_points(merge_df):
    # Read the CSV data
    points_df = pd.read_csv('./csvs/points_table.csv')

    # Iterate through the input arrays
    for team1, team2, inning1_pred, inning2_pred, overs_inning1, overs_inning2 in zip(merge_df['team1'], merge_df['team2'], merge_df['inning1_pred'], merge_df['inning2_pred'],merge_df['overs_inning1'], merge_df['overs_inning2']):
        # Search for the corresponding row in the DataFrame
        if inning1_pred > inning2_pred:
            update_points(points_df, team1, team2, inning1_pred, overs_inning1, inning2_pred, overs_inning2)
        else:
            update_points(points_df, team2, team1, inning2_pred, overs_inning2, inning1_pred, overs_inning1)

    # Sort the DataFrame by points and net run rate
    points_df = points_df.sort_values(by=['Points', 'Net Run Rate'], ascending=False)
    pd.DataFrame.to_csv(points_df, 'updatedTable.csv', index=False)

def mappings(team1, team2="Afghanistan", venue="Narendra Modi Stadium, Ahmedabad"):
    venue_dict = {'Arun Jaitley Stadium, Delhi': 0,
                  'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow': 1,
                  'Eden Gardens, Kolkata': 2,
                  'Himachal Pradesh Cricket Association Stadium, Dharamsala': 3,
                  'M Chinnaswamy Stadium, Bengaluru': 4,
                  'MA Chidambaram Stadium, Chepauk, Chennai': 5,
                  'Maharashtra Cricket Association Stadium, Pune': 6,
                  'Narendra Modi Stadium, Ahmedabad': 7,
                  'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 8,
                  'Wankhede Stadium, Mumbai': 9}
    batting_team_dict = {'Afghanistan': 0,
                         'Australia': 1,
                         'Bangladesh': 2,
                         'England': 3,
                         'India': 4,
                         'Netherlands': 5,
                         'New Zealand': 6,
                         'Pakistan': 7,
                         'South Africa': 8,
                         'Sri Lanka': 9}
    bowling_team_dict = {'Afghanistan': 0,
                         'Australia': 1,
                         'Bangladesh': 2,
                         'England': 3,
                         'India': 4,
                         'Netherlands': 5,
                         'New Zealand': 6,
                         'Pakistan': 7,
                         'South Africa': 8,
                         'Sri Lanka': 9}
    
    # print(team1, team2, venue)

    venue_value = venue_dict[venue]
    team1_value = batting_team_dict[team1]
    team2_value = bowling_team_dict[team2]
    
    return venue_value, team1_value, team2_value

def reverse_mappings(team1_value):
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
    
    # print(team1_value)
    team1_value=batting_team_dict[team1_value]
    
    return team1_value

#base function to predict the match winner
def predict():
    df=pd.read_csv('./csvs/matches.csv')
    df.drop(['season','date', 'match_number','player_of_match', 'umpire1', 'umpire2',
       'reserve_umpire', 'match_referee', 'winner', 'winner_runs',
       'winner_wickets', 'match_type','city'],axis='columns',inplace=True)

    le=LabelEncoder()
    toss_mapping = {}

    for column in df.columns:
        df[column] = le.fit_transform(df[column])
        toss_mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    df=pd.read_csv('./csvs/upcoming_matches.csv')
    number_of_matches=df.shape[0]
    toss_winner=[]
    toss_losser=[]

    number_of_matches = df.shape[0]

    for i in range(0,number_of_matches):
        #generate random number between 0 and 1
        num = np.random.uniform(0, 1)
        # print(num)
        if  num < 0.5:
            toss_winner.append(df.iloc[i]['team1'])
            toss_losser.append(df.iloc[i]['team2'])
        else:
            toss_winner.append(df.iloc[i]['team2'])
            toss_losser.append(df.iloc[i]['team1'])

    df['toss_winner']=toss_winner

    print(df)
    # Use apply and lambda functions to map values based on 'toss_mapping'
    df = df.apply(lambda col: col.map(lambda x: toss_mapping[col.name][x] if col.name in toss_mapping else x))
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df = pd.merge(df, stadium, left_on='venue', right_on='venue', how='left')
    df[['won_after_bat_first', 'won_after_chase', 'first_inning_score', 'second_inning_score']] = toss_scaler.transform(df[['won_after_bat_first', 'won_after_chase', 'first_inning_score', 'second_inning_score']]) 
    toss_prediction = toss_model.predict(df)
    toss_prediction = np.round(toss_prediction)
    print(toss_prediction)
    # toss_prediction = np.array([1 for i in range(0, number_of_matches)])

    batting_team1=[]
    bowling_team1=[]
    venue_x=[]
    Total_Overs_Played=[]
    for i in range(0,number_of_matches):
        if toss_prediction[i]==0:
            batting=toss_winner[i]
            bowling=toss_losser[i]
            batting_team1.append(batting)
            bowling_team1.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)
        else:
            bowling=toss_winner[i]
            batting=toss_losser[i]
            batting_team1.append(batting)
            bowling_team1.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)

    df_inning1=pd.DataFrame()
    df_inning1['venue']=venue_x
    df_inning1['team1']=batting_team1
    df_inning1['team2']=bowling_team1
    df_inning1['total_overs_played']=Total_Overs_Played
    #merge stadium and df_inning1 on venue
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df_inning1 = pd.merge(df_inning1, stadium, left_on='venue', right_on='venue', how='left')
    df_inning1['%']=get_percentages(df_inning1['team1'],df_inning1['team2'])
    df_inning1.drop(['won_after_bat_first', 'won_after_chase', 'second_inning_score'], axis='columns', inplace=True)
    inning1_pred = predict_inning1(df_inning1, toss_mapping, number_of_matches)
    # Over prediction model result
    inning1_pred, overs_inning1 = over_predict_inning1(df_inning1, toss_mapping, number_of_matches, inning1_pred)

    for i in range(len(inning1_pred)):
        if(overs_inning1[i] > 50.0):
            inning1_pred[i] = inning1_pred[i] * 50 / overs_inning1[i]
            overs_inning1[i] = 50.0
    # Inning 2 run prediction model result
    df_inning2 = pd.DataFrame()
    df_inning2['team1'] = bowling_team1
    df_inning2['team2'] = batting_team1
    df_inning2['venue'] = venue_x
    df_inning2['total_overs_played'] = Total_Overs_Played
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df_inning2 = pd.merge(df_inning2, stadium, left_on='venue', right_on='venue', how='left')
    df_inning2.drop(['won_after_bat_first', 'won_after_chase'], axis='columns', inplace=True)
    df_inning2['%'] = get_percentages(df_inning2['team1'], df_inning2['team2'])
    # Inning 2 run prediction model result
    inning2_pred = predict_inning2(df_inning2, toss_mapping, number_of_matches, inning1_pred)

    # over prediction model result for inning 2
    # df_inning2.drop('total_runs_in_innings1', axis='columns', inplace=True)
    inning2_pred, overs_inning2 = over_predict_inning1(df_inning2, toss_mapping, number_of_matches, inning2_pred)

    for i in range(len(inning2_pred)):
        if(overs_inning2[i] > 50.0):
            inning2_pred[i] = inning2_pred[i] * 50 / overs_inning2[i]
            overs_inning2[i] = 50.0

    merge_df = pd.DataFrame()
    merge_df['team1'] = batting_team1
    merge_df['team2'] = bowling_team1
    merge_df['venue'] = venue_x
    merge_df['inning1_pred'] = inning1_pred
    merge_df['inning2_pred'] = inning2_pred
    merge_df['overs_inning1'] = list(overs_inning1)
    merge_df['overs_inning2'] = list(overs_inning2)

    # print(merge_df.head(15))
    winner = []

    for i in range(len(merge_df)):
        if(merge_df['inning1_pred'][i] > merge_df['inning2_pred'][i]):
            winner.append(reverse_mappings(merge_df['team1'][i]))
            print(f"{merge_df['team1'][i]} will win with Batting first with {int(merge_df['inning1_pred'][i])} runs vs {int(merge_df['inning2_pred'][i])} runs {merge_df['team2'][i]} will lose")
        else:
            winner.append(reverse_mappings(merge_df['team2'][i]))
            print(f"{merge_df['team2'][i]} will win with Bowling First with {int(merge_df['inning2_pred'][i])} runs vs {int(merge_df['inning1_pred'][i])} runs {merge_df['team1'][i]} will lose")

    #save this merge_df to csv
    # print(merge_df.head(15))

    final_points(merge_df)

    winner_df = pd.DataFrame({'winner': winner})
    winner_df.to_csv('./csvs/semis_winner.csv', index=False)

def predict_finalists():
    updated_points_df = pd.read_csv('./csvs/updatedTable.csv')

    top_4_df = updated_points_df.head(4)

    teams = top_4_df['Team'].values

    v1,t1, t4 = mappings(teams[0], teams[3], 'Wankhede Stadium, Mumbai')
    v2,t2,t3 = mappings(teams[1], teams[2], 'Eden Gardens, Kolkata')

    data = [
        {'team1':t1, 'team2':t4, 'venue':v1},
        {'team1':t2, 'team2':t3, 'venue':v2}
    ]


    # create a DataFrame from the data
    df = pd.DataFrame(data)

    # write the DataFrame to a CSV file
    df.to_csv('./csvs/upcoming_semis.csv', index=False)

    le=LabelEncoder()
    toss_mapping = {}

    number_of_matches=df.shape[0]
    toss_winner=[]
    toss_losser=[]
    # print(number_of_matches)

    for i in range(0,number_of_matches):
        if np.random.uniform(0, 1) < 0.5:
            toss_winner.append(df.iloc[i]['team1'])
            toss_losser.append(df.iloc[i]['team2'])
        else:
            toss_winner.append(df.iloc[i]['team2'])
            toss_losser.append(df.iloc[i]['team1'])

    df['toss_winner']=toss_winner

    # Use apply and lambda functions to map values based on 'toss_mapping'
    for column in df.columns:
        if column in toss_mapping:
            df[column] = df[column].map(toss_mapping[column])

    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df = pd.merge(df, stadium, left_on='venue', right_on='venue', how='left')
    df[['won_after_bat_first', 'won_after_chase', 'first_inning_score', 'second_inning_score']] = toss_scaler.transform(df[['won_after_bat_first', 'won_after_chase', 'first_inning_score', 'second_inning_score']]) 
    toss_prediction = toss_model.predict(df)
    toss_prediction = np.round(toss_prediction)

    batting_team1=[]
    bowling_team1=[]
    venue_x=[]
    Total_Overs_Played=[]
    for i in range(0,2):
        if toss_prediction[i]==0:
            batting=toss_winner[i]
            bowling=toss_losser[i]
            batting_team1.append(batting)
            bowling_team1.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)
        else:
            bowling=toss_winner[i]
            batting=toss_losser[i]
            batting_team1.append(batting)
            bowling_team1.append(bowling)
            venue_x.append(df.iloc[i]['venue'])
            Total_Overs_Played.append(50.0)

    df_inning1=pd.DataFrame()
    df_inning1['venue']=venue_x
    df_inning1['team1']=batting_team1
    df_inning1['team2']=bowling_team1
    df_inning1['total_overs_played']=Total_Overs_Played
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df_inning1 = pd.merge(df_inning1, stadium, left_on='venue', right_on='venue', how='left')
    df_inning1.drop(['won_after_bat_first', 'won_after_chase', 'second_inning_score'], axis='columns', inplace=True)
    # print(df_inning1['team1'],df_inning1['team2'])
    df_inning1['%']=get_semi_percentages(df_inning1['team1'],df_inning1['team2'])
    inning1_pred = predict_inning1(df_inning1, toss_mapping, number_of_matches)

    inning1_pred, overs_inning1 = over_predict_inning1(df_inning1, toss_mapping, number_of_matches, inning1_pred)

    for i in range(len(inning1_pred)):
        if(overs_inning1[i] > 50.0):
            inning1_pred[i] = inning1_pred[i] * 50 / overs_inning1[i]
            overs_inning1[i] = 50.0

    # Inning 2 run prediction model result
    df_inning2 = pd.DataFrame()
    df_inning2['team1'] = bowling_team1
    df_inning2['team2'] = batting_team1
    df_inning2['venue'] = venue_x
    df_inning2['total_overs_played'] = Total_Overs_Played
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df_inning2 = pd.merge(df_inning2, stadium, left_on='venue', right_on='venue', how='left')
    df_inning2.drop(['won_after_bat_first', 'won_after_chase'], axis='columns', inplace=True)
    df_inning2['%'] = get_semi_percentages(df_inning2['team1'], df_inning2['team2'])

    # Inning 2 run prediction model result
    inning2_pred = predict_inning2(df_inning2, toss_mapping, number_of_matches, inning1_pred)

    # over prediction model result for inning 2
    # df_inning2.drop('total_runs_in_innings1', axis='columns', inplace=True)
    inning2_pred, overs_inning2 = over_predict_inning1(df_inning2, toss_mapping, number_of_matches, inning2_pred)

    for i in range(len(inning2_pred)):
        if(overs_inning2[i] > 50.0):
            inning2_pred[i] = inning2_pred[i] * 50 / overs_inning2[i]
            overs_inning2[i] = 50.0

    merge_df = pd.DataFrame()
    merge_df['team1'] = batting_team1
    merge_df['team2'] = bowling_team1
    merge_df['venue'] = venue_x
    merge_df['inning1_pred'] = list(inning1_pred)
    merge_df['inning2_pred'] = list(inning2_pred)
    merge_df['overs_inning1'] = list(overs_inning1)
    merge_df['overs_inning2'] = list(overs_inning2)

    # print(merge_df.head(15))
    winner = []

    for i in range(len(merge_df)):
        if(merge_df['inning1_pred'][i] > merge_df['inning2_pred'][i]):
            winner.append(reverse_mappings(merge_df['team1'][i]))
            print(f"{reverse_mappings(merge_df['team1'][i])} will win with Batting first with {int(merge_df['inning1_pred'][i])} runs vs {int(merge_df['inning2_pred'][i])} runs {reverse_mappings(merge_df['team2'][i])} will lose")
        else:
            winner.append(reverse_mappings(merge_df['team2'][i]))
            print(f"{reverse_mappings(merge_df['team2'][i])} will win with Bowling First with {int(merge_df['inning2_pred'][i])} runs vs {int(merge_df['inning1_pred'][i])} runs {reverse_mappings(merge_df['team1'][i])} will lose")

    #save this merge_df to csv

    # create a DataFrame from the winner list
    df = pd.DataFrame({'semi_final_winner': winner})
    df.to_csv('./csvs/semis_winner.csv', index=False)

    # print(merge_df.head(15))
    return winner
    
    # Predict for the final

def predict_finalWinner(winner):
    # print(winner)
    v1,t1,t2 = mappings(winner[0], winner[1], 'Narendra Modi Stadium, Ahmedabad')

    data = [
        {'team1':t1, 'team2':t2, 'venue':v1}
    ]

    # create a DataFrame from the data
    df = pd.DataFrame(data)

    # write the DataFrame to a CSV file
    df.to_csv('./csvs/upcoming_final.csv', index=False)

    le=LabelEncoder()
    toss_mapping = {}

    number_of_matches=df.shape[0]
    toss_winner=[]
    toss_losser=[]

    for i in range(0,number_of_matches):
        if np.random.uniform(0, 1) < 0.5:
            toss_winner.append(df.iloc[i]['team1'])
            toss_losser.append(df.iloc[i]['team2'])
        else:
            toss_winner.append(df.iloc[i]['team2'])
            toss_losser.append(df.iloc[i]['team1'])

    df['toss_winner']=toss_winner

    # Use apply and lambda functions to map values based on 'toss_mapping'
    for column in df.columns:
        if column in toss_mapping:
            df[column] = df[column].map(toss_mapping[column])

    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df = pd.merge(df, stadium, left_on='venue', right_on='venue', how='left')
    df[['won_after_bat_first', 'won_after_chase', 'first_inning_score', 'second_inning_score']] = toss_scaler.transform(df[['won_after_bat_first', 'won_after_chase', 'first_inning_score', 'second_inning_score']]) 
    toss_prediction = toss_model.predict(df)
    toss_prediction = np.round(toss_prediction)

    batting_team1=[]
    bowling_team1=[]
    venue_x=[]
    Total_Overs_Played=[]

    if toss_prediction[0]==0:
        batting=toss_winner[0]
        bowling=toss_losser[0]
        batting_team1.append(batting)
        bowling_team1.append(bowling)
        venue_x.append(df.iloc[0]['venue'])
        Total_Overs_Played.append(50.0)
    else:
        bowling=toss_winner[0]
        batting=toss_losser[0]
        batting_team1.append(batting)
        bowling_team1.append(bowling)
        venue_x.append(df.iloc[0]['venue'])
        Total_Overs_Played.append(50.0)


    df_inning1=pd.DataFrame()
    df_inning1['venue']=venue_x
    df_inning1['team1']=batting_team1
    df_inning1['team2']=bowling_team1
    df_inning1['total_overs_played']=Total_Overs_Played
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df_inning1 = pd.merge(df_inning1, stadium, left_on='venue', right_on='venue', how='left')
    df_inning1['%']=get_percentages(df_inning1['team1'],df_inning1['team2'])
    df_inning1.drop(['won_after_bat_first', 'won_after_chase', 'second_inning_score'], axis='columns', inplace=True)
    df_inning1['%']=get_final_percentages(df_inning1['team1'].values[0],df_inning1['team2'].values[0])

    # Inning 1 run prediction model result
    inning1_pred = predict_inning1(df_inning1, toss_mapping, number_of_matches)

    # Over prediction model result
    inning1_pred, overs_inning1 = over_predict_inning1(df_inning1, toss_mapping, number_of_matches, inning1_pred)

    for i in range(len(inning1_pred)):
        if(overs_inning1[i] > 50.0):
            inning1_pred[i] = inning1_pred[i] * 50 / overs_inning1[i]
            overs_inning1[i] = 50.0

    # Inning 2 run prediction model result
    df_inning2 = pd.DataFrame()
    df_inning2['team1'] = bowling_team1
    df_inning2['team2'] = batting_team1
    df_inning2['venue'] = venue_x
    df_inning2['total_overs_played'] = Total_Overs_Played
    stadium = pd.read_csv('./csvs/stadium_details.csv')
    stadium['venue'] = stadium['venue'].str.split(',').str[0]
    stadium['venue'] = le.fit_transform(stadium['venue'])
    df_inning2 = pd.merge(df_inning2, stadium, left_on='venue', right_on='venue', how='left')
    df_inning2.drop(['won_after_bat_first', 'won_after_chase'], axis='columns', inplace=True)
    df_inning2['%'] = get_final_percentages(df_inning2['team1'].values[0], df_inning2['team2'].values[0])

    inning2_pred = predict_inning2(df_inning2, toss_mapping, number_of_matches, inning1_pred)

    inning2_pred, overs_inning2 = over_predict_inning1(df_inning2, toss_mapping, number_of_matches, inning2_pred)

    for i in range(len(inning2_pred)):
        if(overs_inning2[i] > 50.0):
            inning2_pred[i] = inning2_pred[i] * 50 / overs_inning2[i]
            overs_inning2[i] = 50.0

    merge_df = pd.DataFrame()
    merge_df['team1'] = batting_team1
    merge_df['team2'] = bowling_team1
    merge_df['venue'] = venue_x
    merge_df['inning1_pred'] = list(inning1_pred)
    merge_df['inning2_pred'] = list(inning2_pred)
    merge_df['overs_inning1'] = list(overs_inning1)
    merge_df['overs_inning2'] = list(overs_inning2)

    winner = []

    for i in range(len(merge_df)):
        if(merge_df['inning1_pred'][i] > merge_df['inning2_pred'][i]):
            winner.append(reverse_mappings(merge_df['team1'][i]))
            print(f"{reverse_mappings(merge_df['team1'][i])} will win with Batting first with {int(merge_df['inning1_pred'][i])} runs vs {int(merge_df['inning2_pred'][i])} runs {reverse_mappings(merge_df['team2'][i])} will lose")
        else:
            winner.append(reverse_mappings(merge_df['team2'][i]))
            print(f"{reverse_mappings(merge_df['team2'][i])} will win with Bowling First with {int(merge_df['inning2_pred'][i])} runs vs {int(merge_df['inning1_pred'][i])} runs {reverse_mappings(merge_df['team1'][i])} will lose")

    return winner

def semi_winner():
    final_csv = pd.read_csv('./csvs/semis_winner.csv')
    final_csv = final_csv.head()
    return final_csv

def predict_semifinalists():
    final_csv = pd.read_csv('./csvs/updatedTable.csv')
    final_csv = final_csv.head(4)
    return final_csv

###################################################################################################################################
###################################################################################################################################
##########                                  Playing 11's
###################################################################################################################################
###################################################################################################################################

def get_playing11_for_teams():
    #get teams from semi_winner.csv
    semi_winner = pd.read_csv('./csvs/semis_winner.csv')
    semi_winner = semi_winner['semi_final_winner'].values

    team = semi_winner

    print("Finalist")
    print(team)

    df_players = pd.read_csv('./csvs/player_details.csv')

    le = LabelEncoder()

    mapping = {}
    categorical_columns = ['team','player_name']
    for column in categorical_columns:
        df_players[column] = le.fit_transform(df_players[column])
        mapping[column] = dict(zip(le.classes_, le.transform(le.classes_)))

    # print(mapping)

    df_players.drop(['opponent_team', 'venue', 'match_runs', 'match_wickets'], axis = 1, inplace=True)

    # print(df_players)
    
    df_players.drop_duplicates(subset=['player_name'], inplace=True)

    df_players[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']] = player_scaler.fit_transform(df_players[['total_runs','highest_score','batting_avg','bowling_avg','strike_rate','economy','bowling_runs','total_wickets']])

    tmp1, t1_id, tmp2 = mappings(team[0])
    tmp3, t2_id, tmp4 = mappings(team[1])

    team1 = df_players[df_players['team'] == t1_id]
    team2 = df_players[df_players['team'] == t2_id]

    team1['opponent_team'] = t2_id
    team2['opponent_team'] = t1_id

    team1['venue'] = 7
    team2['venue'] = 7

    # print(team1.isnull().sum())

    player_t1 = player_model.predict(team1)
    
    player_t1 = np.round(player_t1)

    player_t1[player_t1 < 0]=0

    player_t1 = pd.DataFrame(player_t1)

    player_t1['id'] = team1['player_name'].values
    
    # print(player_t1)

    player_t2 = player_model.predict(team2)

    player_t2 = np.round(player_t2)

    player_t2[player_t2 < 0]=0

    player_t2 = pd.DataFrame(player_t2)

    player_t2['id'] = team2['player_name'].values

    # print(player_t2)

    player_t1 = player_t1.sort_values(by=[0], ascending=False)

    for i in range(len(player_t1)):

        for key, value in mapping['player_name'].items():
            if value == player_t1['id'][i]:
                player_t1['id'][i] = key

    # print(player_t1)

    batsman_t1 = player_t1.head(6)
    player_t1 = player_t1.sort_values(by=[1], ascending=False)
    bowler_t1 =[]

    player_t1 = player_t1.sort_values(by=[1], ascending=False)

    batsman_t1 = player_t1.head(6)
    batsman_ids = set(batsman_t1['id'])
    bowler_t1 = player_t1[~player_t1['id'].isin(batsman_ids)].head(5)
    
    # playing11_t1 = batsman_t1['id'].values.tolist() + bowler_t1
    #combine batsman_t1 and bowler_t1
    playing11_t1 = batsman_t1['id'].values.tolist() + bowler_t1['id'].values.tolist()

    player_t2 = player_t2.sort_values(by=[0], ascending=False)

    for i in range(len(player_t2)):
        for key, value in mapping['player_name'].items():
            if value == player_t2['id'][i]:
                player_t2['id'][i] = key

    batsman_t2 = player_t2.head(6)
    player_t2 = player_t2.sort_values(by=[1], ascending=False)
    bowler_t2 =[]
    batsman_ids = set(batsman_t2['id'])
    bowler_t2 = player_t2[~player_t2['id'].isin(batsman_ids)].head(5)

    
    playing11_t2 = batsman_t2['id'].values.tolist() + bowler_t2['id'].values.tolist()

    return playing11_t1, playing11_t2, team



###################################################################################################################################
###################################################################################################################################
##########                                  API's
###################################################################################################################################
###################################################################################################################################



@app.get("/")
def index():
    return {"Hello": "World"}

@app.get("/predict")
def predicts():
    winner = predict()
    return {"winner": winner}

@app.get("/predict_semifinalists")
def predict_semifinalist():
    tables = predict_semifinalists()
    return {"tables": tables['Team'].to_dict()}

@app.get("/predict_finalists")
def predict_finalist_team():
    tables = predict_finalists()
    return {"final teams": tables}

@app.get("/predict_winner")
def predict_winner():
    tables = semi_winner()
    print(tables["semi_final_winner"].values)
    winner = predict_finalWinner(tables["semi_final_winner"].values)
    # winner = predict_finalWinner(winner)
    return {"winner": winner}

@app.get("/get_playing11")
def get_playing11():
    team1, team2, teams = get_playing11_for_teams()
    return {teams[0]: team1, teams[1]: team2}

if __name__ == "__main__":
    shutil.copyfile('./csvs/previous_data_per.csv', './csvs/prev_data_per.csv')
    uvicorn.run(app, host="127.0.0.1", port=8000)