### Classification Of Iris Dataset Using Multi-layer Perceptron

#### Setup

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#### Load Data and Prepare Train and Test Set

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5)

#### Build, Train and Test The MLP Network

clf = MLPClassifier(
	hidden_layer_sizes=( 6),
	activation="tanh",
	solver="sgd",
	learning_rate="constant",
	learning_rate_init=1e-2,
	max_iter=2000,
	tol=0,
	momentum=0
)

clf.fit( X_train, y_train)

print( f"Accuracy ( TRAIN ): { clf.score( X_train, y_train)}")
print( f"Accuracy ( TEST ): { clf.score( X_test, y_test)}")