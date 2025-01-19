# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
teams = pd.read_csv("teams.csv")

# Clean the data (Remove extra columns)
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Remove missing values
teams.dropna(inplace=True)

# Split the data into train and test datasets
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# Initialize and train the linear regression model
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train[predictors], train[target])

# Predict the medals for the test dataset
predictions = reg.predict(test[predictors])

# Round predictions and add them to the test dataframe
test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"] = 0  # Ensure no negative predictions
test["predictions"] = test["predictions"].round()

# Calculate mean absolute error of the predictions
error = mean_absolute_error(test["medals"], test["predictions"])
print(f"Mean Absolute Error: {error}")


# Example function to predict medals for a given team and year
def predict_medals(athletes, prev_medals):
    # Create a DataFrame for the new data
    new_data = pd.DataFrame({"athletes": [athletes], "prev_medals": [prev_medals]})

    # Predict the number of medals
    predicted_medals = reg.predict(new_data)
    return round(predicted_medals[0])


# Example: Predict for a specific country and year (e.g., USA in 2016)
athletes = 100  # Example: number of athletes
prev_medals = 3  # Example: number of previous medals

predicted = predict_medals(athletes, prev_medals)
print(f"Predicted number of medals: {predicted}")


# Verify the prediction with actual data
def verify_prediction(team, year, predicted_medals):
    # Find actual medals from the test dataset
    actual_medals = test[(test["team"] == team) & (test["year"] == year)][
        "medals"
    ].values
    if len(actual_medals) > 0:
        actual_medals = actual_medals[0]
        print(f"Actual number of medals for {team} in {year}: {actual_medals}")
        error = abs(actual_medals - predicted_medals)
        print(f"Prediction error: {error} medals")
    else:
        print(f"No data available for {team} in {year}.")


# Example: Verify the prediction for USA in 2016
verify_prediction("IND", 2016, predicted)
