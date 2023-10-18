import pandas as pd
import time
if __name__ == '__main__':
    start_time = time.time()

    # Define your criteria for min ratings per user
    min_reviews_x = 20

    # Define your criteria for min ratings per vine
    min_x_ratings = 100

    # Load the CSV file into a DataFrame
    df = pd.read_csv('XWines_Full_21M_ratings_differentVintagesAveraged.csv')

    # Group the data by UserID and count the number of reviews for each user
    user_review_counts = df['UserID'].value_counts()


    # Filter users based on criteria
    users_with_x_reviews = user_review_counts[user_review_counts >= min_reviews_x].index.tolist()


    # Create separate DataFrames for each user group
    df_x_reviews = df[df['UserID'].isin(users_with_x_reviews)]


    # New variable for dataframe. Next step is reduce the amount of vines in the dataset
    df = df_x_reviews

    # Group the data by WineID and count the number of ratings for each user
    wine_rating_counts = df['WineID'].value_counts()

    # Filter users based on criteria
    wines_with_x_ratings = wine_rating_counts[wine_rating_counts >= min_x_ratings].index.tolist()

    # Create separate DataFrames for each user group
    df_x_ratings = df[df['WineID'].isin(wines_with_x_ratings)]

    # Save the DataFrames to separate CSV files
    df_x_ratings.to_csv("wines_with_min_" + str(min_x_ratings) + "_ratings_and_users_with_min_" + str( min_reviews_x) + "_ratings_FULL.csv", index=False)

    print("Saved the ratings of users that rated at least" + "showing wines with at least " + str(min_x_ratings) + " ratings to CSV file.")
    print("which took %s seconds " % (time.time() - start_time))