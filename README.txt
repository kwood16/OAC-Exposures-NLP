***** Changing directory *****
******************************

'OAC NLP Project' should be the current directory at all times.



***** Storing and loading data *****
************************************

- Store all data files in 'OAC NLP Project\data'
- It should have the following files:
	i.   Metadata with the notes column
	ii.  Training set with binary_adj_goldstd column
	iii. Full dataset
- Change the input and output paths in 'nlp.py' and 'analysis.py' classification and extraction files



***** Running the script *****
******************************

- To run the whole script, type the following commands from the current directory:

chmod a+x run.sh
./run.sh

- To run individual python files, use either of the following commands to run the respective file:

python nlp_concept2.py 
python analysis_extract.py