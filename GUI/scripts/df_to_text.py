import os
import numpy as np
import pandas as pd 

# Initialize dataframe with zeros according to the number of players
def initialize_final_df(csv_file):
  df = pd.read_csv(csv_file)
  final_df = pd.DataFrame()
  for i in range(6):
    final_df[i + 1] = [0] * df.shape[0] 
  return final_df

# Put the values of the first test in the dataframe
def fill_column(csv_file, final_df, checkbox_id):
  df = pd.read_csv(csv_file)
  final_df[int(checkbox_id)] = df.iloc[:,1].values
  return final_df

def assign_test_scores_to_players(final_df, players_index, test_name, test_csv_file):
  test_df = pd.read_csv(test_csv_file)
  players_index = players_index.squeeze()
  for i in range(len(players_index)):
    final_df.at[players_index[i], int(test_name)] = test_df.iloc[np.where(players_index == players_index[i] )[0][0]][1]
  return final_df

# Convert the dataframe to text files for each player
def convert_df_to_txt(final_df):
  i = 0
  for index, row in final_df.iterrows():
      if i > len(final_df):
        break
      else: 
        TEXT_FILE_SAVE_NAME = 'Player_' + str(i)  + '.txt'
        TEXT_FILE_SAVE_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), TEXT_FILE_SAVE_NAME)
        f = open(TEXT_FILE_SAVE_PATH, 'w')
        # To write each value in a seperate line
        for row_value in row.values:
          truncated_row_value = round(row_value, 3)
          f.write(str((truncated_row_value)) + '\n')
        f.close()
        i += 1


# csv_file = r'D:\Downloads\ITI\ITI Graduation Project\Scout4Stars\GUI\data\agility_stats.csv'
# final_df = initialize_final_df(csv_file)
# final_df = fill_column(csv_file, final_df, '5')
# convert_df_to_txt(final_df)