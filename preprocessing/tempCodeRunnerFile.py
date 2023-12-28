# Number of rows per chunk
# chunk_size = 10000

# # Iterate over the DataFrame in chunks of 10,000
# for start in range(0, len(df), chunk_size):
#     # Define the end of the chunk
#     end = min(start + chunk_size, len(df))

#     # Slice the chunk
#     chunk = df.iloc[start:end]

#     # Define the filename for the chunk
#     filename = f'/Users/jaden/Desktop/BU MSMFT/MF703_Programming_for_Finance/Final_project/training_data/no_dim_redu/data_{start}.csv'

#     # Save the chunk to a CSV file
#     chunk.to_csv(filename, index=False)