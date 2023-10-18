import time
from CollaborativeFiltering import CFRecommender


if __name__ == '__main__':
    print("")
    start_time = time.time()
    # Determine accuracy for our rating predictor using as basis:
    # - ratings of 5000 users,
    # - minimum of ratings per user,
    # - minimum of ratings per wine,
    # - parameter p = 5 (amount of similar wines that a user already have rated that should be
                            # included to compute the predicted rating
    # We test the accuracy by adjusting one parameter while fixing the others

    # Testing the amount of ratings that are included:
    print("Part 0:")
    print("")
    print("We test the accuracy on 50000 users without doing further cleaning")

    print("")
    myRecommender = CFRecommender()
    myRecommender.createSimilarityMatrix(
        '100XWines_prototypingDataSet_basedOn_XWines_Full_21M_ratings_differentVintagesAveraged.csv')
    print("100 Users: " + str(round(myRecommender.determineAccuracy(20, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '1000XWines_prototypingDataSet_basedOn_XWines_Full_21M_ratings_differentVintagesAveraged.csv')
    print("1000 Users: " + str(round(myRecommender.determineAccuracy(200, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")

    myRecommender.createSimilarityMatrix(
        '5000XWines_prototypingDataSet_basedOn_XWines_Full_21M_ratings_differentVintagesAveraged.csv')
    print("5000 Users: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")

    myRecommender.createSimilarityMatrix(
        '50000XWines_prototypingDataSet_basedOn_XWines_Full_21M_ratings_differentVintagesAveraged.csv')
    print("50000 Users: " + str(round(myRecommender.determineAccuracy(10000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")


    # Testing the amount of ratings that are included:
    print("Part I:")
    print("")
    print("We test the impact of amount of ratings that are included in our data.")
    print("We consider only the datapoints with at least 20 user ratings")
    print("We only consider wines with at least 20 ratings")
    print("We use p = 5 for the amount of similar wines that should be included in the prediction algorithm")

    print("")
    myRecommender = CFRecommender()
    myRecommender.createSimilarityMatrix('02_Data_amountOfUsers/1000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("1000 Users: " + str(round(myRecommender.determineAccuracy(200, 5, 'A')*100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '02_Data_amountOfUsers/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("5000 Users: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '02_Data_amountOfUsers/25000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("25000 Users: " + str(round(myRecommender.determineAccuracy(5000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '02_Data_amountOfUsers/50000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("50000 Users: " + str(round(myRecommender.determineAccuracy(10000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    print("")
    print("")

    # Testing the impact of assumptions on minimum user ratings:
    print("Part II:")
    print("")
    print("We test the impact of assumptions on minimum user ratings.")
    print("We consider a data set with 5000 users")
    print("We only consider wines with at least 20 ratings")
    print("We use p = 5 for the amount of similar wines that should be included in the prediction algorithm")

    print("")
    myRecommender = CFRecommender()
    myRecommender.createSimilarityMatrix(
        '03_Data_min_user_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_10_ratings_FULL.csv')
    print("Minimum 10 user ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '03_Data_min_user_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("Minimum 20 user ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '03_Data_min_user_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_50_ratings_FULL.csv')
    print("Minimum 50 user ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '03_Data_min_user_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_100_ratings_FULL.csv')
    print("Minimum 100 user ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    print("")
    print("")

    # Testing the impact of assumptions on minimum wine ratings:
    print("Part III:")
    print("")
    print("We test the impact of assumptions on minimum wine ratings.")
    print("We consider a data set with 5000 users")
    print("We only consider users with at least 20 ratings")
    print("We use p = 5 for the amount of similar wines that should be included in the prediction algorithm")

    print("")
    myRecommender = CFRecommender()
    myRecommender.createSimilarityMatrix(
        '04_Data_min_vine_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_10_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("Minimum 10 wine ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '04_Data_min_vine_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("Minimum 20 wine ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '04_Data_min_vine_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_50_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("Minimum 50 wine ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '04_Data_min_vine_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_100_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("Minimum 100 wine ratings: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    print("")
    print("")


    # Testing the impact of assumptions on minimum wine ratings:
    print("Part IV:")
    print("")
    print("We test the impact of parameter p, the amount of similar wines that should be included in the prediction algorithm.")
    print("We consider a data set with 5000 users")
    print("We only consider users with at least 20 ratings")
    print("We use p = 5 for the amount of similar wines that should be included in the prediction algorithm")

    print("")
    myRecommender = CFRecommender()
    myRecommender.createSimilarityMatrix(
        '04_Data_min_vine_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("p = 1: " + str(round(myRecommender.determineAccuracy(1000, 1, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    myRecommender.createSimilarityMatrix(
        '04_Data_min_vine_ratings/5000XWines_prototypingDataSet_basedOn_wines_with_min_20_ratings_and_users_with_min_20_ratings_FULL.csv')
    print("p = 3: " + str(round(myRecommender.determineAccuracy(1000, 3, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    print("p = 5: " + str(round(myRecommender.determineAccuracy(1000, 5, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
    print("p = 10: " + str(round(myRecommender.determineAccuracy(1000, 10, 'A') * 100, 2))
          + "% Average error, including " + str(myRecommender.sparseRating.nnz) + " ratings.")
