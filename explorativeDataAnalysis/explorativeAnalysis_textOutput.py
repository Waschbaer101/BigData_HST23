import pandas as pd
import sys

# Redirect output to a text file
output_file = "EDA_featureDescription.txt"
sys.stdout = open(output_file, "w")

# Load file
input_filename_ratings = "XWines_prototypingDataSet.csv"
rdf = pd.read_csv(input_filename_ratings)

input_filename_wines = "XWines_Full_100K_wines.csv"
wdf = pd.read_csv(input_filename_wines)

print("\nEDA for " + input_filename_ratings + " only Ratings data:")
print(rdf.describe()['Rating'])

print("\nEDA for " + input_filename_wines + " data:")
#wdf_texts = wdf[['Type','Body','Grapes','Harmonize','Acidity','Code']]
wdf_texts = wdf[['Type','Body','Grapes','Acidity','Code']]

print("\n------------------Overview of characteristic of 100 646 wines------------------")
print(wdf_texts.describe())

print("\n------------------Details for each column------------------")
for column in wdf_texts:
    print(wdf[column].value_counts())
    print("")

# Close the output file
sys.stdout.close()

# Restore the original standard output
sys.stdout = sys.__stdout__
