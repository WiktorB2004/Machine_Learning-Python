# Imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Load dataset
iris_dataset = load_iris()

# Extract training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)
# Training model implementation
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# Print model accuracy on testing data
print("Accuracy level for testing data is: {:.2f}".format(knn.score(X_test, y_test)))
