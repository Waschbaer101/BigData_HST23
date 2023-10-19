import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

vis_folder_path = "visualizations/"
dataset_folder_path = "../datasets/"

if not os.path.exists(dataset_folder_path + "meanRating_AmountOfRatings_perUser_FULL.csv"):
    print("meanRating_AmountOfRatings_perWine_FULL.csv not present in folder. Started file creation.")
    # Read the input file into a Pandas DataFrame
    raw_data = pd.read_csv(dataset_folder_path + "XWines_Full_21M_ratings.csv")

    #Create a DataFrame and add the number of ratings to is using a count method
    user_ratings = pd.DataFrame(raw_data.groupby('UserID')['Rating'].mean())
    # Rename the headers
    user_ratings = user_ratings.rename(columns={'Rating': 'Mean Rating'})
    user_ratings['# of ratings'] = raw_data.groupby('UserID')['Rating'].count()

    user_ratings.to_csv(dataset_folder_path + "meanRating_AmountOfRatings_perUser_FULL.csv")

else:
    print("meanRating_AmountOfRatings_perUser_FULL.csv loaded from folder. Skipped file creation.")
    user_ratings = pd.read_csv(dataset_folder_path + "meanRating_AmountOfRatings_perUser_FULL.csv")


sns.set(style="whitegrid")

####################################### Distribution of Mean Rating per User ########################################
# Specify the bin edges to align with rating values
rating_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
bins = [value - 0.25 for value in rating_values] + [5.25]  # Adjust the bin width as needed

plt.hist(user_ratings['Mean Rating'], bins=bins, color='skyblue', edgecolor='black')
plt.xlabel('Mean Rating per User')
plt.ylabel('Amount of Ratings')
plt.xticks(rating_values)  # Set the x-axis ticks to match the specific rating values
#plt.yscale("log")
plt.savefig(vis_folder_path + "Distribution of Mean Rating per User.png")
plt.close()

######################################## Distribution of Amount of Ratings per User ########################################
plt.hist(user_ratings['# of ratings'], color='skyblue', edgecolor='black')
plt.xlabel('Amount of Ratings per Wine')
plt.ylabel('Number of Wines')
plt.yscale("log")
plt.savefig(vis_folder_path + "Distribution of Amount of Ratings per User.png")
plt.close()