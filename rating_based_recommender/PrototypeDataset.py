import pandas as pd
from random import sample
import time

if __name__ == '__main__':
    start_time = time.time()

    # define number of Users
    nUsers = 5000

    # define file names. For input select file with ratings from users with at least 10 ratings
    input_csv_file = 'XWines_Full_21M_ratings_differentVintagesAveraged.csv'
    output_csv_file = str(nUsers) + 'XWines_prototypingDataSet_basedOn_' + input_csv_file

    # import file and analyse
    ratings = pd.read_csv(input_csv_file)
    print('#ratings in selected file: ' + str((len(ratings))))

    # sample nUsers random userIDs
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