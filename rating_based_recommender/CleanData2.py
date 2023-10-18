import pandas as pd

if __name__ == '__main__':
    # Create a DataFrame from the sample data
    original_df = pd.read_csv("XWines_Full_21M_ratings.csv")
    print(original_df.describe())

    # Group by UserID and WineID, then calculate the average rating
    result_df = original_df.groupby(['UserID', 'WineID']).agg({'Rating': ['mean', 'count']}).reset_index()

    # Rename the columns for clarity
    result_df.columns = ['UserID', 'WineID', 'AverageRating', 'AveragedOver#Ratings']

    # Merge the original DataFrame with the combined ratings
    result_df = original_df.drop_duplicates(subset=['UserID', 'WineID']).merge(result_df, on=['UserID', 'WineID'], how='left')

    # Drop the 'Rating' column (the original rating column) and 'Vintage' column
    result_df = result_df.drop('Rating', axis=1)
    result_df = result_df.drop('Vintage', axis=1)

    # Rename the 'AverageRating' column to 'Rating'
    result_df = result_df.rename(columns={'AverageRating': 'Rating'})

    # Print the updated DataFrame
    print(result_df.describe())

    dropped_ratings = len(original_df) - len(result_df)

    # Print the number of dropped ratings
    print(f"Number of ratings dropped: {dropped_ratings}")

    # Save the updated DataFrame to a CSV file
    result_df.to_csv("XWines_Full_21M_ratings_differentVintagesAveraged.csv", index=False)