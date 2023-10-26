import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd
import csv
import time
import pandas_dtype_efficiency as pd_eff
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def handle_listed_categories(wines_df, listName):
    # Remove square brackets and single quotes from "Harmonize" lists
    wines_df[listName] = wines_df[listName].str.replace(r"[\[\]']", "", regex=True)
    # Create binary columns for each unique item
    binary_columns = wines_df[listName].str.get_dummies(sep=", ")
    wines_df = pd.concat([wines_df, binary_columns], axis=1)
    wines_df.drop(listName, axis=1, inplace=True)
    return wines_df


def wines_pre_processing(wines_df):
    ## read wines and drop uninteresting columns
    wines_columns_to_drop = [
        "WineName",
        "Elaborate",
        "Country",
        "Grapes",
        "RegionID",
        "RegionName",
        "WineryID",
        "WineryName",
        "Website",
        "Vintages",
    ]  # List of column names to drop
    wines_df.drop(columns=wines_columns_to_drop, axis=1, inplace=True)

    wines_df = handle_listed_categories(wines_df, "Harmonize")
    # wines_df = handle_listed_categories(wines_df, 'Grapes')
    # Convert cathegorical data into binary vectors
    categorical_features = ["Type", "Body", "Acidity", "Code"]
    vectorized_wine_df = pd.get_dummies(wines_df, columns=categorical_features)
    return vectorized_wine_df


def reviews_data_process(reviews_df):
    ## read reviews and drop uninteresting columns
    reviews_columns_to_drop = [
        "RatingID",
        "Vintage",
        "Date",
    ]  # List of column names to drop
    reviews_df.drop(columns=reviews_columns_to_drop, axis=1, inplace=True)
    return reviews_df


def create_simmilarity_matrix(selected_wine, wines_for_sim_mtx):
    wines_for_sim_mtx = wines_for_sim_mtx.drop("WineID", axis=1)
    print(selected_wine)
    cosine_sim_matrix = cosine_similarity(
        selected_wine.drop("WineID", axis=1), wines_for_sim_mtx
    )
    return cosine_sim_matrix


def optimize_dataframe(frame_to_optimize):
    ## optimize memory usage of matrix
    checker = pd_eff.DataFrameChecker(
        df=frame_to_optimize,
        categorical_threshold=10,  # Optional argument
        float_size=16,  # Optional argument
    )
    checker.identify_possible_improvements()
    optimized_df = checker.cast_dataframe_to_lower_memory_version()

    frame_to_optimize.memory_usage(deep=True)
    optimized_df.memory_usage(deep=True)

    return optimized_df


def process_user_data(reviews_df, wine_vectorized_optimized_df, selected_user):
    ### User data processing ###
    ## group by user
    grouped_reviews = reviews_df.groupby(["UserID"])
    # Select specific user
    singleUser = grouped_reviews.get_group(selected_user)
    ## Find the wines that the user has reviewed
    singleUser_reviewed_Wines = singleUser["WineID"].unique()
    # Create a list of only the wines that the user has reviewed
    filtered_wines_df = wine_vectorized_optimized_df[
        wine_vectorized_optimized_df["WineID"].isin(singleUser_reviewed_Wines)
    ]

    return filtered_wines_df, singleUser


def find_highest_rated_Wine(singleUser, wines_df):
    max_rating = singleUser["Rating"].max()
    top_rated_wines = singleUser[singleUser["Rating"] == max_rating]

    if top_rated_wines.shape[0] > 1:
        random_wine = top_rated_wines.sample(
            1, random_state=42
        )  # Change the random_state as needed for reproducibility
        highest_rated_wine_id = random_wine["WineID"].values[0]
    else:
        highest_rated_wine_id = top_rated_wines["WineID"].values[0]

    highest_rated_wine_index = wines_df.index[
        wines_df["WineID"] == highest_rated_wine_id
    ].tolist()[0]
    return highest_rated_wine_index, highest_rated_wine_id


def predict_on_closest_wine_sim_nn(
    wine_vectorized_optimized_df, singleUser, model, sampled_wines, wines_df
):
    ## Sample n wines to select similar wines from

    # sim matrix
    sim_matrix = create_simmilarity_matrix(sampled_wines)

    ## PCA for sampled wines
    pca = PCA(n_components=6)
    pca.fit(sampled_wines.drop("WineID", axis=1))
    transformed_wine = pca.transform(sampled_wines.drop("WineID", axis=1))

    # find highest rated wine id and index
    reference_wine_index, highest_rated_wine_id = find_highest_rated_Wine(
        singleUser, wines_df
    )

    ## use similarity matrix to find closest fit to highest scored wine
    similarity_scores = sim_matrix[reference_wine_index]
    most_similar_wine_index = similarity_scores.argmax()
    highest_rated_wine_features = wine_vectorized_optimized_df[
        wine_vectorized_optimized_df["WineID"] == highest_rated_wine_id
    ].values
    most_similar_wine_by_sim_matrix = wine_vectorized_optimized_df.iloc[
        [most_similar_wine_index]
    ]

    ## nearest neighbor to find closest wine
    highest_rated_wine_features_pca = pca.transform(
        [highest_rated_wine_features[0][1:]]
    )
    nearest_neighbor = NearestNeighbors(n_neighbors=20)
    nearest_neighbor.fit(sampled_wines)
    distancies, indices = nearest_neighbor.kneighbors(
        highest_rated_wine_features, n_neighbors=1
    )
    similar_wines_NN = sampled_wines.iloc[indices[0]]

    ## predict by nearestNeighbors
    print("NN", model.predict(similar_wines_NN.drop(["WineID"], axis=1)))

    ## predict by similarity matrix
    print(
        "Similarity matrix",
        model.predict(most_similar_wine_by_sim_matrix.drop(["WineID"], axis=1)),
    )


def sim_matrix_get_similar(
    sim_matrix,
    reference_wine_index,
    highest_rated_wine_id,
    wine_vectorized_optimized_df,
):
    ## use similarity matrix to find closest fit to highest scored wine
    similarity_scores = sim_matrix[reference_wine_index]
    most_similar_wine_index = similarity_scores.argmax()
    highest_rated_wine_features = wine_vectorized_optimized_df[
        wine_vectorized_optimized_df["WineID"] == highest_rated_wine_id
    ].values
    most_similar_wine_by_sim_matrix = wine_vectorized_optimized_df.iloc[
        [most_similar_wine_index]
    ]
    return most_similar_wine_by_sim_matrix


def new_sim_matrix_get(sim_matrix, highest_rated_wine_id, Wine_vectorized_optimized_df):
    highest_sim_index = np.argmax(sim_matrix)
    highest_sim_wine_id = Wine_vectorized_optimized_df.iloc[highest_sim_index]["WineID"]
    highest_sim_wine_features = Wine_vectorized_optimized_df[
        Wine_vectorized_optimized_df["WineID"] == highest_sim_wine_id
    ]
    return highest_sim_wine_features


def get_X_Y_4user(reviews_df, wine_vectorized_optimized_df, UserID):
    filtered_wines_df, singleUser = process_user_data(
        reviews_df, wine_vectorized_optimized_df, UserID
    )

    # Sort both lists according to wineID so that the reviews and wines line up
    singleUser = singleUser.sort_values("WineID")
    user_filtered_wines_df = filtered_wines_df.sort_values("WineID")

    ## fit data to linear regression model
    merged_data = filtered_wines_df.merge(singleUser, on="WineID", how="inner")

    X = merged_data.drop(["Rating", "UserID", "WineID"], axis=1)
    Y = merged_data["Rating"]

    return X, Y, user_filtered_wines_df, singleUser


"""
This funtion tests the accuracy of the regression model
"""


def test_regression_model(reviews_csv_name):
    reviews_csv = reviews_csv_name
    wine_data_csv = "xWines_full_100k_wines.csv"
    ## wines data processing ##
    wines_df = pd.read_csv(wine_data_csv)
    vectorized_wine_df = wines_pre_processing(wines_df)
    wine_vectorized_optimized_df = optimize_dataframe(vectorized_wine_df)
    ### User data processing ###
    reviews_df = pd.read_csv(reviews_csv)
    reviews_df = reviews_data_process(reviews_df)

    user_sample_size = 1000
    sampled_users_df = reviews_df.sample(n=user_sample_size, random_state=8)
    mean_absolute_errors = 0
    proportional_errors = 0

    for UserID in sampled_users_df["UserID"].unique():
        ## filter list of wines user have reviewed
        filtered_wines_df, singleUser = process_user_data(
            reviews_df, wine_vectorized_optimized_df, UserID
        )

        # Sort both lists according to wineID so that the reviews and wines line up
        singleUser = singleUser.sort_values("WineID")
        filtered_wines_df = filtered_wines_df.sort_values("WineID")

        ## fit data to linear regression model
        merged_data = filtered_wines_df.merge(singleUser, on="WineID", how="inner")

        X = merged_data.drop(["Rating", "UserID", "WineID"], axis=1)
        Y = merged_data["Rating"]

        ## train test split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        model = Ridge()
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        proportional_errors += proportional_error_calc(y_pred, Y_test)
        mean_absolute_errors += mean_absolute_error(Y_test, y_pred)

    print("mean proportional error: ", proportional_errors / user_sample_size)
    print("mean absolute errors average: ", (mean_absolute_errors / user_sample_size))


def proportional_error_calc(y_pred, y_test):
    proportional_errors = []
    for pred, true in zip(y_pred, y_test):
        if true != 0:
            error = (abs((pred - true) / true)) * 100
            proportional_errors.append(error)
    # Calculate the mean proportional error
    mean_proportional_error = np.mean(proportional_errors)
    return mean_proportional_error


def wine_centered_recommender():
    ## CSV names
    reviews_csv = "XWines_prototypingDataSet_basedOn_users_with_10_reviews_FULL.csv"
    wine_data_csv = "xWines_full_100k_wines.csv"
    ## wines data processing ##
    wines_df = pd.read_csv(wine_data_csv)
    vectorized_wine_df = wines_pre_processing(wines_df)
    wine_vectorized_optimized_df = optimize_dataframe(vectorized_wine_df)
    ### User data processing ###
    reviews_df = pd.read_csv(reviews_csv)
    reviews_df = reviews_data_process(reviews_df)

    ## sample a random user
    User = reviews_df.sample(n=1, random_state=42)
    print(int(User.iloc[0]["UserID"]))
    ## Relevant data filtering for the user
    x, y, user_filtered_wines, single_user = get_X_Y_4user(
        reviews_df, wine_vectorized_optimized_df, int(User.iloc[0]["UserID"])
    )
    ### train model for the user
    model = Ridge()
    model.fit(x, y)
    ## get the best wine according to the user
    highest_rated_wine_index, highest_rated_wine_id = find_highest_rated_Wine(
        single_user, wines_df
    )
    ## Sample wines from the 100k wines list
    highest_rated_wine_features = wine_vectorized_optimized_df[
        wine_vectorized_optimized_df["WineID"] == highest_rated_wine_id
    ]
    """    highest_rated_wine_features = wine_vectorized_optimized_df.iloc[
        highest_rated_wine_index
    ]
    """
    print(highest_rated_wine_features)
    ## get similar wines from similarity matrix of the n sampled wines
    sim_matrix = create_simmilarity_matrix(
        highest_rated_wine_features, wine_vectorized_optimized_df
    )
    most_similar_wine = new_sim_matrix_get(
        sim_matrix, highest_rated_wine_id, wine_vectorized_optimized_df
    )
    ## Score the wines
    recommended_wine_score = model.predict(most_similar_wine.drop(["WineID"], axis=1))
    ## recommend the wine with the highest score
    print(
        "The recommended wine for ",
        User["UserID"].values,
        " is: ",
        most_similar_wine["WineID"].values,
        "With score: ",
        recommended_wine_score,
    )

    # predict_on_closest_wine_sim_nn(wine_vectorized_optimized_df, singleUser)


if __name__ == "__main__":
    ## Log time for runtime calculation
    start_time = time.time()

    ## Would be cool to recommend a white, a red, and a sparkling for the user

    #### Run the recommender ####
    # wine_centered_recommender()
    #############################

    ############### test accuracy of linear regression ###############
    reviews_csv_name = (
        "XWines_prototypingDataSet_basedOn_users_with_50_reviews_FULL.csv"
    )
    test_regression_model(reviews_csv_name)
    ##################################################################

    ## Calc and print total time to execute
    print("This took %.2f seconds " % (time.time() - start_time))
