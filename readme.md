# CP03 Data Diggers
## ICC World Cup 2023 Predictions
## Team ID : 10

## Contributors : Vrund Rajput (202001075), Hetav Vakani (202001447), Om Patel (202001462)

In this project we had to mainly work on three tasks
Task 1 : Predict Highest run scorer and highest wicket taker
Task 2 : Predict the finalist teams and playing 11
Task 3 : Predict the Winner of ICC Cricket World Cup 2023

## Deployed API : https://dm-cp3.onrender.com

## Task 1 :- Predict the Highest run scorer and the highest wicket taker

For predicting the highest runs scorer/highest wicket taker  what we have done is took the top 10 runs scorer/top 10 wicket taker from the previous played matches and then for each of the top 10 players we predict the runs scored/wickets taken (Run_Wicket_Predictor.ipynb) and then we add these predicted runs/wickets to their current runs/wickets to get the top player.

### Results

Top Scoring Batsman

![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/1dc264cd-7407-46c2-965c-487a5eca7068)


Top Wicket Takers

![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/4f2c0609-013e-441e-a18c-0546566d9cfc)


## Task 2 :- Predict the finalist teams and playing 11

We had the data of the 32 matches played and hence to predict the finalists we needed to predict the winners of the rest of the matches. For predicting the winner of a particular match what we are doing is firstly we predict the toss winner and their decision (bat or bowl) using Toss_Winner_Predictor.ipynb. Then we predict the first innings score (Inning1_Run_Predictor.ipynb) and the overs took by the batting team to score the given runs (Over_Predictor.ipynb). Then we predict the runs scored by the second team (Inning1_Run_Predictor.ipynb) and the respective overs they took to score the runs (Over_Predictor.ipynb). Now as we have the runs scored by both the teams we can decide the winner. The reason we are predicting the overs is to calculate the net run rate so that if there is a tie in points between two teams we can decide the top four teams through net run rate. Now we have the top 4 teams and as before we complete the match between 1st and 4th team and the 2nd and 3rd team to get the finalist teams.

To predict the playing 11 what we have done is we predicted the runs and wickets of the all the palyers belonging to the team.Then we pick the first 6 run scorer and keep picking the most wicket takers untill the team size is 11.

### Result of the remaining Matches
![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/1b447a76-2283-4761-9686-0d39ddb743a5)

### Semifinalists
![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/110fb594-1e5f-47ec-8e0e-7b8996d20912)

### Finalists
![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/dcd51c6f-aaa6-4c07-99da-3d9405d3c89e)


### Playing Eleven of the finalists
![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/0bf3e073-039e-4f6c-8c52-789f2369d010)


## Task 3 :- Predict the Winner of ICC Cricket World Cup 2023

From task 2 we have the finalists and to predict the winner we complete the match using the similar process used before to get the winner of the finals. 

### Winner
![image](https://github.com/Vrund87/ICC-Cricket-World-Cup-2023/assets/75675477/f092c5dc-1e20-4df1-b6d0-3857446c3a07)

