import numpy as np
from sklearn.preprocessing import Imputer
import argparse
import csv
import pickle

NUCLEOTIDEDICT = {'A':0, 'C':1, 'G':2, 'T':3}
TARGETGENEDICT = {}
DRUGGENEDICT = {}
def read_file(fileName):
	data = []
	with open(fileName,"rb") as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
				row = list(row)[1:]
				#row = map(float,row)
				data.append(row)	
	return np.array(data)

def parse_sequence(seq):
	numNucleotides = len(seq)
	parsedSeq = []
	for i in range(numNucleotides):
		parsedSeq.append(NUCLEOTIDEDICT[seq[i]])
	return parsedSeq

def parse_target_gene(targetGene):
	if not TARGETGENEDICT.has_key(targetGene):
		numTargetGenes = len(TARGETGENEDICT)
		TARGETGENEDICT[targetGene] = numTargetGenes
	return TARGETGENEDICT[targetGene]

def parse_drug_gene(drugGene):
	if not DRUGGENEDICT.has_key(drugGene):
		numDrugGenes = len(DRUGGENEDICT)
		DRUGGENEDICT[drugGene] = numDrugGenes
	return DRUGGENEDICT[drugGene]

def parse_crispr_data(data):
	numObjects = len(data)
	numNucleotides = len(data[0][0])
	newData = []
	for i in range(numObjects):
		tempNewData = parse_sequence(data[i][0])
		tempNewData.append(parse_target_gene(data[i][1]))
		tempNewData += map(float,data[i][2:6])
		tempNewData.append(parse_drug_gene(data[i][7]))
		tempNewData.append(float(data[i][-1]))
		newData.append(tempNewData)
	return np.array(newData)

def pickle_data(data,fileName):
	with open(fileName[:-4]+'.pkl', 'wb') as f:
		pickle.dump(data,f)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', type=str, help="Crispr dataset", required=True)
	args = parser.parse_args()
	fileName = args.f
	data = read_file(fileName)
	data = parse_crispr_data(data)
	print(data.shape)

if __name__ == "__main__":
	main()
