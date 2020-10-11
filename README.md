# Accelerometer-data-Prediction
The given accelerometer data was classified into walking, standing, going up stairs, going down the stairs.

Methodology ---
Step 1: Import data from CSV files. The data is characterized into ID, Timestamps, UTC time, X,Y,Z values.
Step 2: Modify data by engineering features like avg displacement, velocity and acceleration.   
Step 3: Define Models - Forest Classifier, Logistic Regression and SVM
Step 4: Test each Model using cross validation and feature importances using L1 and L2 regularization
Step 5: Use the best model and features to test accuracy of the validation set and predict the values of the new data.

Results - All movements were walking with a maximum accuracy of 93%. However, this could be due to overtraining as there is not enough data. 
