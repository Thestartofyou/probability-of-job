from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare your data: X represents the features (qualifications) and y represents the job outcome (getting the job or not)
X = ...  # Your feature data
y = ...  # Your target variable (job outcome)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train your model
model = LogisticRegression()  # or RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the probability of getting the job based on your qualifications
your_qualifications = ...  # Your qualifications (should match the features used to train the model)
probability = model.predict_proba([your_qualifications])[0][1]
print("Probability of getting the job:", probability)

