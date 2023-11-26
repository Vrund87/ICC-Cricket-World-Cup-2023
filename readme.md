# CP03 Data Diggers
## ICC World Cup 2023 Predictions
## Team ID : 10

In this project we had to mainly work on three tasks
Task 1 : Predict Highest run scorer and highest wicket taker
Task 2 : Predict the finalist teams and playing 11
Task 3 : Predict the Winner of ICC Cricket World Cup 2023

## Task 1 :- Predict the Highest run scorer and the highest wicket taker

For predicting the highest runs scorer/highest wicket taker  what we have done is took the top 10 runs scorer/top 10 wicket taker from the previous played matches and then for each of the top 10 players we predict the runs scored/wickets taken (Run_Wicket_Predictor.ipynb) and then we add these predicted runs/wickets to their current runs/wickets to get the top player.

## Task 2 :- Predict the finalist teams and playing 11

We had the data of the 32 matches played and hence to predict the finalists we needed to predict the winners of the rest of the matches. For predicting the winner of a particular match what we are doing is firstly we predict the toss winner and their decision (bat or bowl) using Toss_Winner_Predictor.ipynb. Then we predict the first innings score (Inning1_Run_Predictor.ipynb) and the overs took by the batting team to score the given runs (Over_Predictor.ipynb). Then we predict the runs scored by the second team (Inning1_Run_Predictor.ipynb) and the respective overs they took to score the runs (Over_Predictor.ipynb). Now as we have the runs scored by both the teams we can decide the winner. The reason we are predicting the overs is to calculate the net run rate so that if there is a tie in points between two teams we can decide the top four teams through net run rate. Now we have the top 4 teams and as before we complete the match between 1st and 4th team and the 2nd and 3rd team to get the finalist teams.

To predict the playing 11 what we have done is we predicted the runs and wickets of the all the palyers belonging to the team.Then we pick the first 6 run scorer and keep picking the most wicket takers untill the team size is 11.

## Task 3 :- Predict the Winner of ICC Cricket World Cup 2023

From task 2 we have the finalists and to predict the winner we complete the match using the similar process used before to get the winner of the finals. 
