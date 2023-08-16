package com.LSTMPredictor.mavenproject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class NNApp 
{
	public static void main(String[] args) {
		// Load and preprocess data from .csv file

		// Convert data to DataSetIterator
//				DataSet dataSet = new DataSet(features, labels);
//				DataSet dataSet = getTrainCsv("data.csv");
		DataSet dataSet = getTrain();
		DataSet evalData = getEvaluate();

		// Normalize the data using MinMaxScaler
		NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
		normalizer.fitLabel(true);
		normalizer.fit(dataSet);
		normalizer.transform(dataSet);
		normalizer.fit(evalData);
		normalizer.transform(evalData);
//				normalizer.transform(labels);

		
		DataSetIterator iterator = new ExistingDataSetIterator(dataSet);
		DataSetIterator evaluationIterator = new ExistingDataSetIterator(evalData);

// Build and train the LSTM network with an additional layer
		int batchSize = 5;
		int numInputs = 2;
		int lstmLayerSize = 50;
		int additionalLayerSize = 30; // Size of the additional hidden layer
		int numOutputs = 1;
		int numEpochs = 5;

		MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder().seed(123)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(0.01)).list()
				.layer(0, new LSTM.Builder()
						.nIn(numInputs)
						.nOut(lstmLayerSize)
						.activation(Activation.TANH)
						.build())
				.layer(1, new DenseLayer.Builder()
						.nIn(lstmLayerSize)
						.nOut(additionalLayerSize)
						.activation(Activation.RELU)
						.build())
//						.layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
								.activation(Activation.IDENTITY)
								.nIn(additionalLayerSize) // Input size from the
								.nOut(numOutputs).build())
				.build();

		MultiLayerNetwork network = new MultiLayerNetwork(builder);
		network.init();
		network.setListeners(new ScoreIterationListener(10), new EvaluativeListener(evaluationIterator, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
//		for (int i = 0; i < numEpochs; i++) {
//			iterator.reset();
////			iterator.next(batchSize);
		network.fit(iterator, numEpochs);
//		}

		// Create an Evaluation object
//		        Evaluation evaluation = new Evaluation();
//		Evaluation evaluation = new Evaluation();
//
//		// Iterate through the evaluation dataset and make predictions
//		while (evaluationIterator.hasNext()) {
//			DataSet evaluationData = evaluationIterator.next();
//			INDArray evalFeatures = evaluationData.getFeatures();
//			INDArray evalLabels = evaluationData.getLabels();
//
//			// Make predictions using the trained network
//			INDArray predictions = network.output(evalFeatures, false);
//
//			// Evaluate the predictions against the labels
//			evaluation.eval(evalLabels, predictions);
//		}
//
//		// Print the evaluation metrics
//		System.out.println(evaluation.stats());
	}
	
	private static DataSet getTrainCsv(String path) {
		// TODO Auto-generated method stub
		String filePath = path;
		try {
			filePath = new ClassPathResource("data.csv").getFile().getPath();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try (RecordReader recordReader = new CSVRecordReader(';')) {
			recordReader.initialize(new FileSplit(new java.io.File(filePath)));

			List<double[]> dataList = new ArrayList<>();
			List<double[]> labelList = new ArrayList<>();

			while (recordReader.hasNext()) {
				List<Writable> record = recordReader.next();
				double[] values = record.stream().limit(record.size() - 1) // Exclude the last value
						// (ExpectedMovement)
						.mapToDouble(writable -> Double.parseDouble(writable.toString())).toArray();

				double expectedMovement = Double.parseDouble(record.get(record.size() - 1).toString());

				dataList.add(values);
				labelList.add(new double[] { expectedMovement });
			}

			System.out.println(dataList.toString());
			System.out.println(labelList.toString());

			INDArray features = Nd4j.create(dataList.toArray(new double[0][0]));
			INDArray labels = Nd4j.create(labelList.toArray(new double[0][0]));

			return new DataSet(features, labels);
		} catch (Exception e) {
			return null;
		}
	}

	public static DataSet getTrain() {
	        double[][][] inputArray = {
//	            {{18.7}, {181}},
//	            {{17.4}, {186}},
//	            {{18}, {195}},
//	            {{19.3}, {193}},
//	            {{20.6}, {190}},
//	            {{17.8}, {181}},
//	            {{19.6}, {195}},
//	            {{18.1}, {193}},
//	            {{20.2}, {190}},
//	            {{17.1}, {186}},
	            {{1}, {1}},
	            {{2}, {2}},
	            {{3}, {3}},
	            {{4}, {4}},
	            {{5}, {5}},
	            {{6}, {6}},
	            {{7}, {7}},
	            {{8}, {8}},
	            {{9}, {9}},
	            {{10}, {10}},
	
	        };
	        
	        double[][] outputArray = {
//	                {3750},
//	                {3800},
//	                {3250},
//	                {3450},
//	                {3650},
//	                {3625},
//	                {4675},
//	                {3475},
//	                {4250},
//	                {3300},
	        		{2},
	        		{4},
	        		{6},
	        		{8},
	        		{10},
	        		{12},
	        		{14},
	        		{16},
	        		{18},
	        		{20},
	     
	        };
	        
	        INDArray input = Nd4j.create(inputArray);
	        INDArray labels = Nd4j.create(outputArray);
	        
	        return new DataSet(input, labels);
	    }
	 
	 public static DataSet getEvaluate() {
	        double[][][] inputArray = {
//	            {{18.7}, {181}},
//	            {{17.4}, {186}},
//	            {{18}, {195}},
//	            {{19.3}, {193}},
//	            {{15}, {190}},
//	            {{17.8}, {181}},
//	            {{19.6}, {195}},
//	            {{56}, {205}},
//	            {{20.2}, {190}},
//	            {{17.1}, {186}},
	            {{18}, {18}},
	            {{2}, {2}},
	            {{3}, {3}},
	            {{4}, {4}},
	            {{5}, {5}},
	            {{6}, {6}},
	            {{7}, {7}},
	            {{63}, {63}},
	            {{52}, {52}},
	            {{135}, {135}},
	        };
	        
	        double[][] outputArray = {
//	                {3750},
//	                {3800},
//	                {3250},
//	                {3450},
//	                {3650},
//	                {3625},
//	                {4675},
//	                {3475},
//	                {4250},
//	                {3300},
	          		{36},
	        		{4},
	        		{6},
	        		{8},
	        		{10},
	        		{12},
	        		{14},
	        		{126},
	        		{104},
	        		{270},
	        };
	        
	        INDArray input = Nd4j.create(inputArray);
	        INDArray labels = Nd4j.create(outputArray);
	        
	        return new DataSet(input, labels);
	 }
}
