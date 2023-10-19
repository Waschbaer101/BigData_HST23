# Source: https://www.nickmccullum.com/python-machine-learning/recommendation-systems-python/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

vis_folder_path = "visualizations/"
dataset_folder_path = "../datasets/"

if not os.path.exists(dataset_folder_path + "meanRating_AmountOfRatings_perWine_FULL.csv"):
    print("meanRating_AmountOfRatings_perWine_FULL.csv not present in folder. Started file creation.")
    #Import the data
    rating_data = pd.read_csv(dataset_folder_path + "XWines_prototypingDataSet.csv") 
    #rating_data = pd.read_csv(dataset_folder_path + "XWines_Full_21M_ratings.csv") 
    wine_data = pd.read_csv(dataset_folder_path + "XWines_Full_100K_wines.csv") 

    #Merge our two data sources
    merged_data = pd.merge(rating_data, wine_data, on='WineID')

    #Calculate aggregate data
    merged_data.groupby('WineName')['Rating'].mean().sort_values(ascending = False)
    merged_data.groupby('WineName')['Rating'].count().sort_values(ascending = False)

    #Create a DataFrame and add the number of ratings to is using a count method
    ratings_data = pd.DataFrame(merged_data.groupby('WineName')['Rating'].mean())
    # Rename the headers
    ratings_data = ratings_data.rename(columns={'Rating': 'Mean Rating'})
    ratings_data['# of ratings'] = merged_data.groupby('WineName')['Rating'].count()

    ratings_data.to_csv(dataset_folder_path + "meanRating_AmountOfRatings_perWine_FULL.csv")

else:
    print("meanRating_AmountOfRatings_perWine_FULL.csv loaded from folder. Skipped file creation.")
    ratings_data = pd.read_csv(dataset_folder_path + "meanRating_AmountOfRatings_perWine_FULL.csv")


#Make some visualizations
sns.set(style="whitegrid")

# Histograms
######################################## Distribution of Mean Rating per Wine ########################################
# Specify the bin edges to align with rating values
rating_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
bins = [value - 0.25 for value in rating_values] + [5.25]  # Adjust the bin width as needed

plt.hist(ratings_data['Mean Rating'], bins=bins, color='skyblue', edgecolor='black')
plt.xlabel('Mean Rating per Wine')
plt.ylabel('Amount of Wines')
plt.xticks(rating_values)  # Set the x-axis ticks to match the specific rating values
plt.savefig(vis_folder_path + "Distribution of Mean Rating per Wine.png")
plt.close()

######################################## Distribution of Amount of Ratings per Wine ########################################
plt.hist(ratings_data['# of ratings'], color='skyblue', edgecolor='black')
plt.xlabel('Amount of Ratings per Wine')
plt.ylabel('Number of Wines')
plt.yscale("log")
plt.savefig(vis_folder_path + "Distribution of Amount of Ratings per Wine.png")
plt.close()

# Jointplot
if not os.path.exists(vis_folder_path + "Mean Rating to Amount of Ratings per Wine.png"):
    print("in creation: Mean Rating to Amount of Ratings per Wine.png")

    # Create a joint plot using Seaborn
    jointplot = sns.jointplot(data=ratings_data, x='Mean Rating', y='# of ratings', color='skyblue', edgecolor='black')
    jointplot.ax_joint.set_yscale('log')

    # Set labels and titles using Seaborn style
    jointplot.set_axis_labels('Mean Rating per Wine', 'Amount of Ratings per Wine')

    # Save the plot
    jointplot.savefig(vis_folder_path + 'Mean Rating to Amount of Ratings per Wine.png')

else:
    print("Mean Rating to Amount of Ratings per Wine.png already exists. Skipped file creation.")