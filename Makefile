# the executable for Python
PYTHON=python3
# the directory the main framework exists in
MAIN=neural_stylization
# the directory with the dataset
DATASET=${MAIN}/dataset
# the build directory
BUILD=build
# the pointer to the pdflatex binary
PDFLATEX=pdflatex
# the pointer to the bibilography engine (bibtex)
BIBTEX=bibtex

# install Python dependencies in the requirements.txt
install:
	${PYTHON} -m pip install -r requirements.txt

# make a build directory
build:
	mkdir -p ${BUILD}

# make the presentation
presentation: build
	cd tex && pdflatex presentation
	cp tex/presentation.pdf ${BUILD}

# make the report without references
report: build
	cd tex && ${PDFLATEX} report
	cp tex/report.pdf ${BUILD}

# make the report using bibtex to compile references
report_w_ref: build
	cd tex && \
		${PDFLATEX} report && \
		${BIBTEX} report && \
		${PDFLATEX} report && \
		${PDFLATEX} report
	cp tex/report.pdf ${BUILD}
