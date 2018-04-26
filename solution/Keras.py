import time

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

print(time.time())
train_df = pd.read_csv('../dump/train_modified.csv')
test_df = pd.read_csv('../dump/test_modified.csv')

target = 'acc_now_delinq'
IDcol = 'member_id'
train_df[target].value_counts()
predictors = [x for x in train_df.columns if x not in [target, IDcol]]

# Split Train/Test data
x_train, x_valid, y_train, y_valid = train_test_split(train_df[predictors], train_df[target], test_size=0.15, random_state=14)

# Create the model
model = Sequential()

# Define the three layered model
model.add(Dense(110, input_dim=68, kernel_initializer="uniform", activation="relu"))
model.add(Dense(110, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, epochs=22000, batch_size=200)

performance = model.evaluate(x_valid, y_valid)
print("%s: %.2f%%" % (model.metrics_names[1], performance[1] * 100))

# Predict using the trained model
prediction = model.predict(test_df[predictors])
rounded_predictions = [round(x) for x in prediction]
print(rounded_predictions)
