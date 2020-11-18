import sys
import math
import numpy as np 
import _pickle as pickle
sys.path.append("../tools/")
import pickle
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit

### Depickle the pickle file and load it into the dictionary enron_data
enron_data = pickle.load(open("D:\ML Project\Enron Case\enron_data.pkl", "rb"))

### Print the number of people in the Enron dictionary
print("No of People in the Enron Dataset", len(enron_data))

### Print the features of the nested dictionary
first_item = list(enron_data.keys())[0]
print(enron_data[first_item].keys())

### Print the number of POI in the enron_data
people = 0
poiCount = 0
for person in enron_data:
    people = people + 1
    if enron_data[person]['poi'] == 1:
        poiCount += 1
print("No of POI:", poiCount)

### Print the number of POI in the text file
poiFile = open("D:\ML Project\Enron Case\poi_names.txt")
poiCount2 = 0
lines = poiFile.readlines() 
for line in lines:
    if line[0] == '(':
        poiCount2 += 1
print("Total POIs:",poiCount2)

### Data Exploarion on different persons in the enron_data dictionary
jeffrey_k_skilling = enron_data['SKILLING JEFFREY K']
print(jeffrey_k_skilling)
print("Skilling's Salary:", enron_data["SKILLING JEFFREY K"]["salary"])
print("Skilling's Bonus", enron_data["SKILLING JEFFREY K"]["bonus"])
print("Skilling's Total Stocks Value", enron_data["SKILLING JEFFREY K"]["total_stock_value"])
print("Lay's Salary:", enron_data["LAY KENNETH L"]["salary"])
print("Lay's Bonus", enron_data["LAY KENNETH L"]["bonus"])
print("Lay's Total Stocks Value", enron_data["LAY KENNETH L"]["total_stock_value"])

### Printing the number of people havig quantified salary
people = 0
have_quantified_salary = 0
for person in enron_data:
    people = people + 1
    if enron_data[person]['salary'] != 'NaN':
            have_quantified_salary += 1
print(have_quantified_salary)

### Plotting Salary Histogram
feature1 = ["salary"]
enron_data.pop('TOTAL', 0)
salary = featureFormat(enron_data, feature1)
counts, bins = np.histogram(salary)
plt.hist(bins[:-1], bins, weights=counts)

### Printing outliers in Salary array
for x in salary:
    if x>1000000 :
        print("Outlier:", x)
for person in enron_data:
    if enron_data[person]["salary"] == 1072321:
        print(person)
    if enron_data[person]["salary"] == 1111258:
        print(person)
    if enron_data[person]["salary"] == 1060932:
        print(person)

### Plotting Bonus Histogram
feature2 = ["bonus"]
enron_data.pop('TOTAL', 0)
bonus = featureFormat(enron_data, feature2)
counts, bins = np.histogram(bonus)
plt.hist(bins[:-1], bins, weights=counts)

### Printing outliers in Bonus array
for x in bonus:
    if x>5000000 :
        print("Outlier:", x)
for person in enron_data:
    if enron_data[person]["bonus"] == 8000000:
        print(person)
    if enron_data[person]["bonus"] == 7000000:
        print(person)
    if enron_data[person]["bonus"] == 5249999:
        print(person)
    if enron_data[person]["bonus"] == 5600000:
        print(person)

### Finding POIs from the salary and bonus outliers
if enron_data["LAVORATO JOHN J"]["poi"] == 1:
    print("John J Lavorato is a Person of Interest (POI)")
if enron_data["LAY KENNETH L"]["poi"] == 1:
    print("Kenneth L Lay is a Person of Interest (POI)")
if enron_data["BELDEN TIMOTHY N"]["poi"] == 1:
    print("Timothy N Belden is a Person of Interest (POI)")
if enron_data["SKILLING JEFFREY K"]["poi"] == 1:
    print("Jeffrey K Skilling is a Person of Interest (POI)")

### Plotting Scatterplot for Data Visualization
enron_data.pop('TOTAL', 0)
features_list = ["bonus", "salary"]
data = featureFormat(enron_data, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### Using Sklearn train test split to divide data into trainig and testing data
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"

### Building the Regression Model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(feature_train, target_train)
coef = reg.coef_
intercept = reg.intercept_
print('Slope: {}, Intercept: {}'.format(coef, intercept))
training_score = reg.score(feature_train, target_train)
test_score = reg.score(feature_test, target_test)
print('Score when same data is used to train and test: {}'.format(training_score))
print('Score when separate test data is used: {}'.format(test_score))

for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 
### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

### Drawing the Regression Line on the Scatter PLot
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
reg.fit(feature_test, target_test)
new_coef = reg.coef_
print('Slope of new regression line fitted on test data: {}'.format(new_coef))
plt.plot(feature_train, reg.predict(feature_train), color="b") 
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()