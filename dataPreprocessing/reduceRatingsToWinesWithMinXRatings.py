import pandas as pd
import time

start_time = time.time()

# Define your criteria
min_x_ratings = 50

# Load the CSV file into a DataFrame
df = pd.read_csv('XWines_Full_21M_ratings.csv')

# Group the data by UserID and count the number of ratings for each user
wine_rating_counts = df['WineID'].value_counts()

# Filter users based on criteria
wines_with_min_x_ratings = wine_rating_counts[wine_rating_counts >= min_x_ratings].index.tolist()

# Create separate DataFrames for each user group
df_ratings_from_wines_with_min_x_ratings = df[df['WineID'].isin(wines_with_min_x_ratings)]

# Save the DataFrames to separate CSV files
df_ratings_from_wines_with_min_x_ratings.to_csv("ratings_from_wines_with_min_" + str(min_x_ratings) + "_ratings_FULL.csv", index=False)

print("Saved the ratings of users from wines with at least " + str(min_x_ratings) + " ratings to CSV file.")
print("which took %s seconds " % (time.time() - start_time))