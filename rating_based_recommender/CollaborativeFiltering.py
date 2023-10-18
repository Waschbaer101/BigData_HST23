from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sparse
import pandas as pd
from scipy.sparse import coo_matrix
import time

class CFRecommender:
    def __init__(self, similarityMatrix = None, newUserIDs = None, newWineIDs = None,
                            originalUserIDs = None, originalWineIDs = None, sparseRating = None):
        self.similarityMatrix = similarityMatrix
        self.newUserIDs = newUserIDs
        self.newWineIDs = newWineIDs
        self.originalUserIDs = originalUserIDs
        self.originalWineIDs = originalWineIDs
        self.sparseRating = sparseRating
        self.testSparseRatingTraining = None
        self.testSparseRatingTest = None


    def createSimilarityMatrix(self, filename):
        # import file and prepare data
        df = pd.read_csv(filename)
        num_users = df['UserID'].nunique()
        num_wines = df['WineID'].nunique()

        user_indices = []
        wine_indices = []
        ratings = []

        # create arrays with connected user_id-s, wine_id-s and ratings at the same indices
        for _, row in df.iterrows():
            user_id = row['UserID']
            wine_id = row['WineID']
            rating = row['Rating']
            user_indices.append(user_id)
            wine_indices.append(wine_id)
            ratings.append(rating)

        # Transformate IDs in order to create coo matrix with IDs as rows/columns
        self.newUserIDs = {user_id: index for index, user_id in enumerate(np.unique(user_indices))}
        self.newWineIDs = {wine_id: index for index, wine_id in enumerate(np.unique(wine_indices))}
        new_user_indices = [self.newUserIDs[user_id] for user_id in user_indices]
        new_wine_indices = [self.newWineIDs[wine_id] for wine_id in wine_indices]

        # Safe old IDs
        self.originalUserIDs = {index: user_id for user_id, index in self.newUserIDs.items()}
        self.originalWineIDs = {index: user_id for user_id, index in self.newWineIDs.items()}

        # Create matrix with user_id-s as rows, wine_id-s as columns and ratings as input
        ratingMatrix = coo_matrix((ratings, (new_user_indices, new_wine_indices)), shape=(num_users, num_wines),
                                  dtype=np.float32)

        # Transform into sparse matrix
        self.sparseRating = sparse.csr_matrix(ratingMatrix)

        # HelpVariable
        helpSparse = self.sparseRating.copy()

        # Compute rating average of a user's non-zero entries
        non_zero_entries = helpSparse.data  # Alle Nicht-Null-Einträge
        average = non_zero_entries.mean()

        # Substitute user's rating average from user's ratings
        helpSparse.data -= average

        # Calculate Cosine Similarity
        self.similarityMatrix = cosine_similarity(helpSparse.T)

    # Predict rating for a given wine (old ID) and ratingDictionary
    def computeRatingOldWineID(self, usersRatings, wineID):

        #Transform old wineID to the new ID in our dictionnary and similarityMatrix
        newWineID = self.newWineIDs.get(wineID, -1)

        # Copy dictionary of usersRatings and replace ratings by similarity value to newWineID
        similarityDictionary = {}
        for item in usersRatings:
            similarityDictionary[item] = self.similarityMatrix[newWineID][item]

        # Take only the five vineIDs with the highest similarity to our newWineID
        sortedSimilarities = dict(sorted(similarityDictionary.items(), key=lambda item: item[1], reverse=True))
        topFiveWines = list(sortedSimilarities.keys())[:5]

        #Compute the average rating of the five similarst wines of the user's already rated wines
        sumRating = 0
        for i in range(len(topFiveWines)):
            sumRating += usersRatings[topFiveWines[i]]
        newRating = sumRating/5

        return newRating

    # Predict rating for a given wine (new ID) and ratingDictionary
    def computeRatingNewWineID(self, usersRatings, wineID, includedSimilarWines):
        # Copy dictionary of usersRatings and replace ratings by similarity value to newWineID
        similarityDictionary = {}
        for item in usersRatings:
            similarityDictionary[item] = self.similarityMatrix[wineID][item]

        # Take only the #includedSimilarWines vineIDs with the highest similarity to our newWineID
        sortedSimilarities = dict(sorted(similarityDictionary.items(), key=lambda item: item[1], reverse=True))
        topSimilarWines = list(sortedSimilarities.keys())[:includedSimilarWines]

        # Compute the average rating of the five similarst wines of the user's already rated wines
        sumRating = 0
        for i in range(len(topSimilarWines)):
            sumRating += usersRatings[topSimilarWines[i]]
        newRating = sumRating / includedSimilarWines

        return newRating

    def computeRatingWeightedNewWineID(self, usersRatings, wineID, includedSimilarWines):
        # Copy dictionary of usersRatings and replace ratings by similarity value to newWineID
        similarityDictionary = {}
        for item in usersRatings:
            similarityDictionary[item] = self.similarityMatrix[wineID][item]

        # Take only the #includedSimilarWines vineIDs with the highest similarity to our newWineID
        sortedSimilarities = dict(sorted(similarityDictionary.items(), key=lambda item: item[1], reverse=True))
        topSimilarWines = list(sortedSimilarities.keys())[:includedSimilarWines]
        topSimilarWinesSimilarities = list(sortedSimilarities.values())[:includedSimilarWines]

        # Compute the average rating of the five similarst wines of the user's already rated wines
        sumRating = 0
        sumSimilarities = 0
        for i in range(len(topSimilarWines)):
            sumRating += (usersRatings[topSimilarWines[i]]*topSimilarWinesSimilarities[i])
            sumSimilarities += topSimilarWinesSimilarities[i]
        if (sumSimilarities == 0):
            newRating = sum(usersRatings.values())/len(usersRatings.values())
        else:
            newRating = sumRating / sumSimilarities

        return newRating

    # Get for a user (oldID) a dictionary of his ratings with the new Wine-IDs
    def userRatingsOldID(self, userID):
        newUserID = self.newUserIDs.get(userID, -1)
        userRatings = {}
        for col_index in self.sparseRating.getrow(newUserID).indices:
            userRatings[col_index] = self.sparseRating[newUserID, col_index]
        return userRatings

    # Get for a user (newID) a dictionary of his ratings with the new Wine-IDs
    def userRatingsNewID(self, newUserID):
        userRatings = {}
        for col_index in self.sparseRating.getrow(newUserID).indices:
            userRatings[col_index] = self.sparseRating[newUserID, col_index]
        return userRatings

    # Give a array of number of amount similar wines using the new wineID as input, and new wineID as output
    def helpSimilarWinesNewID(self, newWineID, amount):
        similar_items = np.argsort(self.similarityMatrix[newWineID])[::-1]
        top_similar_items = similar_items[1:amount + 1]
        return top_similar_items

    # Give a array of number of amount similar wines using the old wineID as input, and new wineID as output
    def helpSimilarWinesOldID(self, oldWineID, amount):
        newWineID = self.newWineIDs.get(oldWineID, -1)
        similar_items = np.argsort(self.similarityMatrix[newWineID])[::-1]
        top_similar_items = similar_items[1:amount + 1]
        return top_similar_items

    # Give a recommendation of 10 wines based on our Rating model
    def recommendWine(self, usersRatings, includedWines, similarwines, amountToRecommend):
        newRatedWines = {}
        # Compute average rating of user
        for item in usersRatings:
            # Only consider user's rated wines that are rated above the average
            tempSimilarWines = self.helpSimilarWinesNewID(item, similarwines)
            for similarWine in tempSimilarWines:
                # only consider wines that the user have not already rated
                if usersRatings.get(similarWine) is None:
                    # Predict wine rating with the model of function "computeRatingNewWineID"
                    tempRating = self.computeRatingNewWineID(usersRatings, similarWine, includedWines)
                    newRatedWines[similarWine] = tempRating
        # return the 10 best rated wines of our predictions
        recommendations = dict(sorted(newRatedWines.items(), key=lambda item: item[1], reverse=True))
        return (dict(list(recommendations.items())[:amountToRecommend]))


    # Divide Data into training and test data
    def generateTrainingData(self, sizeTestData):
        length = self.sparseRating.shape[0] - sizeTestData
        self.testSparseRatingTraining = self.sparseRating[:length, :]
        self.testSparseRatingTest = self.sparseRating[length:, :]

    def generateNewRecommender(self):
        helpSparse = self.testSparseRatingTraining.copy()

        # Compute rating average of a user's non-zero entries
        non_zero_entries = helpSparse.data  # Alle Nicht-Null-Einträge
        average = non_zero_entries.mean()

        # Substitute user's rating average from user's ratings
        helpSparse.data -= average

        # Calculate Cosine Similarity
        newSimilarityMatrix = cosine_similarity(helpSparse.T)
        newUserIDs = self.newUserIDs.copy()
        newWineIDs = self.newWineIDs.copy()
        originalUserIDs = self.originalUserIDs.copy()
        originalWineIDs = self.originalWineIDs.copy()
        sparseRating = self.testSparseRatingTraining.copy()

        return CFRecommender(newSimilarityMatrix, newUserIDs, newWineIDs,
                             originalUserIDs, originalWineIDs, sparseRating)


    # Compute Accuracy of ratings based on training and test data
    def determineAccuracy(self, sizeOfTestdata, sizeOfWineIncluded, WeightedOrAverage):
        self.generateTrainingData(sizeOfTestdata)
        newRecommender = self.generateNewRecommender()
        sizeTrainingsdata = self.testSparseRatingTraining.shape[0]
        n = 0
        sum = 0
        for i in range(sizeOfTestdata):
            userratings = self.userRatingsNewID(sizeTrainingsdata + i)
            for item in userratings:
                tempdict = userratings.copy()
                tempdict.pop(item)
                tempRating = userratings[item]
                if (WeightedOrAverage == 'A'):
                    predictedRating = newRecommender.computeRatingNewWineID(tempdict, item, sizeOfWineIncluded)
                    tempaccuracy = abs((predictedRating-tempRating) / tempRating)
                elif (WeightedOrAverage == 'W'):
                    predictedRating = newRecommender.computeRatingWeightedNewWineID(tempdict, item, sizeOfWineIncluded)
                    tempaccuracy = abs((predictedRating - tempRating) / tempRating)
                else:
                    return print("Use 'A' or 'W' as input")

                n += 1
                sum += tempaccuracy
        accuracy = sum / n
        return accuracy








# Get the nearest neighbors for an item (e.g., item 0)
# item_id = 17186
# similar_items = np.argsort(item_similarity[item_id])[::-1]

# Recommendations for item 0 based on its nearest neighbors
# recommendations = [item for item in similar_items if item != item_id]
