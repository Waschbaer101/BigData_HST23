import pandas as pd
import time

start_time = time.time()

# Define your criteria
min_x_ratings = 50

# Load the CSV file into a DataFrame
df = pd.read_csv('XWines_Full_21M_ratings.csv')

# Group the data by UserID and count the number of ratings for each user
user_rating_counts = df['UserID'].value_counts()

# Filter users based on criteria
users_with_min_x_ratings = user_rating_counts[user_rating_counts >= min_x_ratings].index.tolist()

# Create separate DataFrames for each user group
df_ratings_from_users_with_min_x_ratings = df[df['UserID'].isin(users_with_min_x_ratings)]

# Save the DataFrames to separate CSV files
df_ratings_from_users_with_min_x_ratings.to_csv("ratings_from_users_with_min_" + str(min_x_ratings) + "_ratings_FULL.csv", index=False)

print("Saved the ratings of users with at least " + str(min_x_ratings) + " ratings to CSV file.")
print("which took %s seconds " % (time.time() - start_time))

# 130.31525492668152 seconds for full dataset