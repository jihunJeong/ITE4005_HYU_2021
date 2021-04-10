from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train_df = pd.read_table('./train/dt_train1.txt', sep='\s+')
train_df['car_evaluation'].replace(('unacc', 'acc', 'good', 'vgood'), (0, 1, 2, 3), inplace = True)
train_df['lug_boot'].replace(('small', 'med', 'big'), (0, 1, 2), inplace = True)
train_df['safety'].replace(('low', 'med', 'high'), (0, 1, 2), inplace = True)
train_df['buying'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace = True)
train_df['maint'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace = True)
train_df['doors'].replace('5more', 5, inplace = True)
train_df['persons'].replace('more', 5, inplace = True)

x_train = train_df.iloc[:,:6]
y_train = train_df.iloc[:, 6]

test_df = pd.read_table('./test/dt_answer1.txt', sep='\s+')
test_df['car_evaluation'].replace(('unacc', 'acc', 'good', 'vgood'), (0, 1, 2, 3), inplace = True)
test_df['lug_boot'].replace(('small', 'med', 'big'), (0, 1, 2), inplace = True)
test_df['safety'].replace(('low', 'med', 'high'), (0, 1, 2), inplace = True)
test_df['buying'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace = True)
test_df['maint'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace = True)
test_df['doors'].replace('5more', 5, inplace = True)
test_df['persons'].replace('more', 5, inplace = True)
x_test = test_df.iloc[:,:6]
y_test = test_df.iloc[:,6]

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_test: ", y_test.shape)

model = RandomForestClassifier()
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# creating a model
model = DecisionTreeClassifier(max_depth = 10)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))
