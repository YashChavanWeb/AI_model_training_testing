from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Initialize the Flask app
app = Flask(__name__)

# Load and preprocess the data
teams = pd.read_csv("teams.csv")
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
teams.dropna(inplace=True)

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# Train the linear regression model
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train[predictors], train[target])


# Function to predict medals
def predict_medals(athletes, prev_medals):
    new_data = pd.DataFrame({"athletes": [athletes], "prev_medals": [prev_medals]})
    predicted_medals = reg.predict(new_data)
    return round(predicted_medals[0])


# Function to verify prediction and return actual result
def verify_prediction(team, year, predicted_medals):
    actual_medals = test[(test["team"] == team) & (test["year"] == year)][
        "medals"
    ].values
    if len(actual_medals) > 0:
        actual_medals = actual_medals[0]
        error = abs(actual_medals - predicted_medals)
        return actual_medals, error
    return None, None


# Home route that renders the form
@app.route("/")
def home():
    return render_template("index.html")


# Route to handle form submission and prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get values from the form
    team = request.form["team"]
    country = request.form["country"]
    year = int(request.form["year"])
    athletes = int(request.form["athletes"])
    age = int(request.form["age"])
    prev_medals = int(request.form["prev_medals"])

    # Make the prediction
    predicted_medals = predict_medals(athletes, prev_medals)

    # Verify the prediction by getting the actual data
    actual_medals, error = verify_prediction(team, year, predicted_medals)

    # Return the result
    return render_template(
        "index.html",
        prediction=predicted_medals,
        actual_medals=actual_medals,
        error=error,
        team=team,
        year=year,
    )


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
