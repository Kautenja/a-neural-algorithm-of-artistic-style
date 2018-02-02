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

# make a build directory
build:
	mkdir -p build

# make the presentation
presentation: build
	cd tex && pdflatex presentation
	cp tex/presentation.pdf build

# make the report using bibtex to compile references
report: build
	cd tex && \
		pdflatex report && \
		bibtex report && \
		pdflatex report && \
		pdflatex report
	cp tex/report.pdf build
