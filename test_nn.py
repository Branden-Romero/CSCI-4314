import network
import preprocess
import data
import metrics
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', type=str, help="Crispr dataset", required=True)
	args = parser.parse_args()
	
	fileName = args.f
	rawData = preprocess.read_file(fileName)
	crisprData = preprocess.parse_crispr_data(rawData)
	sets = data.Data(crisprData,.4)

	neuralNet = network.Network([36,20,10,5,1])
	neuralNet.SGD(sets.train, 1000, 32, .1, test_data=sets.test)
		
if __name__ == "__main__":
	main()
