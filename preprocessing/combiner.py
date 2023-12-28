import pandas as pd
import os
from datetime import datetime, timedelta

class Combiner:
    def __init__(self, base_path):
        self.base_path = base_path

    def read_file(self, file_path, file_type):
        df = pd.read_csv(file_path)
        # Drop 'Unnamed: 0' column if it exists
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # Preserving 'DateTime' column while adding prefixes to other columns
        df = df.rename(columns={col: f'{file_type}_{col}' for col in df.columns if col != 'DateTime'})
        return df

    def process_daily_data(self, date):
        date_str = date.strftime('%Y-%m-%d')
        file_types = ['ASK', 'BID_ASK', 'BID', 'IV', 'MIDPOINT', 'TRADES', 'Volatility']
        dataframes = {}

        for file_type in file_types:
            file_path = os.path.join(self.base_path, date_str + '_' + file_type + '.csv')
            if os.path.exists(file_path):
                dataframes[file_type] = self.read_file(file_path, file_type)
            else:
                return None  # File not found for this day

        if len(dataframes) < len(file_types):
            return None  # Not all files found for this day

        # Adjust merging logic to account for 'DateTime' column
        merged_df = dataframes['ASK']
        for key in ['BID_ASK', 'BID', 'IV', 'MIDPOINT', 'TRADES']:
            if key in dataframes:
                # Ensuring 'DateTime' is a common column before merging
                merged_df = pd.merge(merged_df, dataframes[key], on='DateTime', how='left')

        return merged_df

    def process_year_data(self, start_date, end_date):
        all_data = []
        for current_date in pd.date_range(start_date, end_date):
            daily_data = self.process_daily_data(current_date)
            if daily_data is not None:
                all_data.append(daily_data)
    
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()



















# import pandas as pd
# import glob

# class Combiner:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.combined_df = None

#     def read_and_combine_csv(self):
#         # Use glob to get all the csv files in the directory
#         csv_files = glob.glob(self.file_path)

#         # List to hold data from each CSV file
#         dataframes = []

#         # Loop through all files and read them into a dataframe
#         for file in csv_files:
#             df = pd.read_csv(file)
#             dataframes.append(df)

#         # Concatenate all dataframes into one
#         self.combined_df = pd.concat(dataframes, ignore_index=True)

#         # Remove any unwanted columns, if necessary
#         self.combined_df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

#         return self.combined_df

#     def save_combined_csv(self, output_path):
#         if self.combined_df is not None:
#             self.combined_df.to_csv(output_path)
#         else:
#             print("No combined dataframe to save. Please run read_and_combine_csv first.")
