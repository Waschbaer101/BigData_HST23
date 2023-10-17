import pandas as pd
from random import sample
import time

start_time = time.time()

# define file names. For input select file with ratings from users with at least 10 ratings
input_csv_file = 'users_with_10_reviews_FULL.csv'
output_csv_file = 'XWines_prototypingDataSet_basedOn_' + input_csv_file

# import file and analyse
ratings = pd.read_csv(input_csv_file)
print('#ratings in selected file: ' + str((len(ratings))))

# sample 10 000 random userIDs
nUsers = 10000
userIDs = ratings['UserID'].unique()
randomSampledUserIDs = sample(list(userIDs), nUsers)
print('#unique userIDs in selected file: ' + str(len(userIDs)))
print('')

# extract all ratings from those userIDs
filteredRatings = ratings[ratings['UserID'].isin(randomSampledUserIDs)]

# export prototyping data set
filteredRatings.to_csv(output_csv_file, index=False)

print("Created prototyping data set from " + str(nUsers) + " random UserIDs. Resulting in " + str(len(filteredRatings)) + " ratings.")
print("This took %.2f seconds " % (time.time() - start_time))