# Define the Python files
TRAIN_SCRIPT=code/SVMTrainTestSplit.py
TEST_SCRIPT=code/testing.py

VENV=venv

# Default target
all: install

# Create a virtual environment and install all dependencies
install:
	pip install pandas matplotlib seaborn plotly scikit-learn joblib numpy

# Train the model
train:
	python $(TRAIN_SCRIPT)

# Test the model
test:
	python $(TEST_SCRIPT)

venv:
	python -m venv $(VENV)
	. $(VENV)/bin/activate; pip install pandas matplotlib seaborn plotly scikit-learn joblib numpy
