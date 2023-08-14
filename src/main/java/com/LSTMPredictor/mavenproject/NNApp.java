package com.LSTMPredictor.mavenproject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
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
		String filePath;
		try {
			filePath = new ClassPathResource("data.csv").getFile().getPath();
			try (RecordReader recordReader = new CSVRecordReader(';')) {
				recordReader.initialize(new FileSplit(new java.io.File(filePath)));

				List<double[]> dataList = new ArrayList<>();
				List<double[]> labelList = new ArrayList<>();

				while (recordReader.hasNext()) {
					List<Writable> record = recordReader.next();
					if (record.size() <= 1) {
						double[] values = record.stream().limit(record.size() - 1) // Exclude the last value
																					// (ExpectedMovement)
								.mapToDouble(writable -> Double.parseDouble(writable.toString())).toArray();

						dataList.add(values);
					} else {
						double expectedMovement = Double.parseDouble(record.get(record.size() - 1).toString());
						labelList.add(new double[] { expectedMovement });
					}
				}
				
				System.out.println(dataList.toString());
				System.out.println(labelList.toString());

				INDArray features = Nd4j.create(dataList.toArray(new double[0][0]));
				INDArray labels = Nd4j.create(labelList.toArray(new double[0][0]));
				
				// Convert data to DataSetIterator
				DataSet dataSet = new DataSet(features, labels);
				DataSetIterator iterator = new ExistingDataSetIterator(dataSet);

				// Normalize the data using MinMaxScaler
				NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
				normalizer.fitLabel(true);
				normalizer.fit(dataSet);
				normalizer.transform(features);
//				normalizer.transform(labels);

				

// Build and train the LSTM network with an additional layer
				int numInputs = 2;
				int lstmLayerSize = 50;
				int additionalLayerSize = 30; // Size of the additional hidden layer
				int numOutputs = 1;
				int numEpochs = 10;

				MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder().seed(123)
						.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
						.weightInit(WeightInit.XAVIER).updater(new Adam(0.01)).list()
						.layer(new LSTM.Builder().nIn(numInputs).nOut(lstmLayerSize).activation(Activation.TANH)
								.build())
						.layer(new DenseLayer.Builder().nIn(lstmLayerSize).nOut(additionalLayerSize)
								.activation(Activation.RELU).build())
						.layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
								.activation(Activation.IDENTITY).nIn(additionalLayerSize) // Input size from the
																							// additional layer
								.nOut(numOutputs).build())
						.build();

				MultiLayerNetwork network = new MultiLayerNetwork(builder);
				network.init();

				for (int i = 0; i < numEpochs; i++) {
					iterator.reset();
					network.fit(iterator);
				}
			} catch (NumberFormatException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
