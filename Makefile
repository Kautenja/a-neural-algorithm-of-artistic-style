# the executable for pip
PIP=pip3
# the executable for Python
PYTHON=python3
# the directory the main framework exists in
MAIN=neural-stylization
# the directory with the dataset
DATASET=${MAIN}/dataset

# install Python dependencies in the requirements.txt
install:
	${PIP} install -r requirements.txt

# download the dataset for the project
download_data:
	cd ${DATASET} && make all

# automatically detect and executes test cases in the main framework
test:
	${PYTHON} -m unittest discover ${MAIN}
