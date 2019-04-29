#!/bin/bash

echo "installing virtualenv"
pip install virtualenv
if ! [ -d "venv" ]; then
	echo "creating new virtual environment for this project..."
	python -m virtualenv venv

	echo "activating the newly created virtualenv..."
	source venv/bin/activate

	echo "installing libraries..."
	python -m pip install -U pandas pyConTextNLP spacy scikit-learn
	python -m spacy download en
else:
	echo "virtual environment already exits... activating it..."
	source venv/bin/activate
fi

echo "executing NLP Extraction..."
python nlp_concept2.py
echo "completed NLP Extraction..."
echo "executing NLP Classification..."
python nlp.py

echo "executing analysis..."
python analysis_extract.py
