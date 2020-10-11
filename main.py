import numpy as np, pandas as pd
import time as t
import project
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

start_time = t.perf_counter()

## Import the respective datasets as pandas dataframe
train_times = pd.read_csv ("C:/Users/ShreyasMisra/Desktop/python/ML/accelerometer_project/train_time_series.csv")
train_labels = pd.read_csv("C:/Users/ShreyasMisra/Desktop/python/ML/accelerometer_project/train_labels.csv")
test_times = pd.read_csv("C:/Users/ShreyasMisra/Desktop/python/ML/accelerometer_project/test_time_series.csv")
test_labels = pd.read_csv("C:/Users/ShreyasMisra/Desktop/python/ML/accelerometer_project/test_labels.csv")
del test_labels["label"]

# Modifies data using the functions from the 'Project.py' file
project.modify_data(train_times, train_labels)
project.modify_data(test_times,test_labels)

# Drop unnecessary data
train_labels = train_labels.replace([np.inf, -np.inf], np.nan)
train_labels = train_labels.dropna(how="any")

test_labels = test_labels.replace([np.inf, -np.inf], np.nan)
test_labels = test_labels.dropna(how="any")

# All features were tested and these were found to be the best ones with high accuracy
features = ['x_avg','y_avg',"z_avg"]
target = "label"

# Training data
X = np.array(train_labels[features])
y = np.array(train_labels[target])

# Model. Forest Classifier was found to produce the best results
forest_classifier = RandomForestClassifier(n_estimators = 100,bootstrap=True,n_jobs=-1)


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75)
forest_classifier.fit(X_train,y_train)
accuracy_forest_clf = forest_classifier.score(X_test,y_test)


print('accuracy of forest classifier: ',accuracy_forest_clf,'\n') # max acc= 93.3%

# feature importances
# print(sorted(list(zip(features,forest_classifier.feature_importances_)) ,key = lambda x: x[1]))

# testing data
predictors = np.array(test_labels[features])
predictions = forest_classifier.predict(predictors)

test_labels["predictions"] = predictions
test_labels.to_csv("test_predictions.csv")

# Predicting data
predictions = dict(Counter(predictions))
print(predictions)

end_time = t.perf_counter()
print("elapsed time: ", end_time-start_time)
