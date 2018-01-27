# the executable for pip
PIP=pip3
# the executable for Python
PYTHON=python3
# the directory the main framework exists in
MAIN=neural_stylization
# the directory with the dataset
DATASET=${MAIN}/dataset

# install Python dependencies in the requirements.txt
install:
	${PIP} install -r requirements.txt

# automatically detect and executes test cases in the main framework
test:
	${PYTHON} -m unittest discover ${MAIN}
