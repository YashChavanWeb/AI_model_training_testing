# How to create a AI Model

1. Form a Hypothesis

- We can predict how many medals a country can win in the olympics

2. Find the data

- get the dataset with rows and columns

3. Reshape the data

- this is applicable when,
  - the target column isn't available in the single row
  - the predicting columns are not available in a single row

4. Clean the data

- there can be missing values in the columns
- most machine learning algorithms cannot work with missing data

5. Find the error metric

- This is to evaluate the performance of our ML Model
- our model will make predictions and they will be different from what is the actual one
- we need to figure out the predictions are good or not
- Mean absolute error: i = 1 to D |xi - yi| -> our prediction - actual value

6. Split the data

- Train the data on the first part
- and then test the data on the remaining part
- so that the model will not have the entire information and we can train the model

7. Train the Model

- Use linear regression (popular machine learning model)
- Y = ax + B (Single variable linear regression)
- This draws a line between the x and the y axis (the linear regression line)
- helps us predict new data using the past data
- the line is what we have trained and we can use it to predict future data
- Y = a1x1 + a2x2 + B (use two factors for training the model)

pip install -r requirements.txt
