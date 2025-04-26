# Define the Python files
TRAIN_SCRIPT=random_forest.py
TEST_SCRIPT=testing.py

VENV=venv

# Default target
all: install

# Create a virtual environment and install all dependencies
install:
	pip install pandas matplotlib seaborn plotly scikit-learn joblib numpy

# Train the model
train:
	cd code && python $(TRAIN_SCRIPT)

# Test the model
test:
	cd code && python $(TEST_SCRIPT)

venv:
	python -m venv $(VENV)
	. $(VENV)/bin/activate; pip install pandas matplotlib seaborn plotly scikit-learn joblib numpy
