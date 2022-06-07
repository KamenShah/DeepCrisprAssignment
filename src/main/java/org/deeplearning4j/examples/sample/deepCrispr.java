///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.deeplearning4j:deeplearning4j-core:1.0.0-M2
//DEPS org.nd4j:nd4j-native:1.0.0-M2
/* *****************************************************************************
 *
 * This is an implementation of DeepCrispr in DeepLearning4j
 * Data is read in through numpy files using the ND4J library
 * The model is then trained and evaluated
 * Visualization can be seen at http://localhost:9000
 ******************************************************************************/
package org.deeplearning4j.examples.sample;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.slf4j.Logger;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.nd4j.evaluation.classification.ROC;


import java.io.File;


public class deepCrispr {
    private static final Logger log = LoggerFactory.getLogger(deepCrispr.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64; // Batch size
        int nEpochs = 10
		; // Number of training epochs
        int seed = 123; // random seed

        /*
            Create an iterator using the batch size for one iteration
         */
        log.info("Load data....");
        INDArray train_X = Nd4j.createFromNpyFile(new File("./train_X.npy"));
        INDArray train_y = Nd4j.createFromNpyFile(new File("./train_y.npy"));
        INDArray test_X = Nd4j.createFromNpyFile(new File("./test_X.npy"));
        INDArray test_y = Nd4j.createFromNpyFile(new File("./test_y.npy"));

        final DataSet allDataTrain = new DataSet(train_X, train_y);
        final DataSet allDataTest = new DataSet(test_X, test_y);

        final List<DataSet> trainList = allDataTrain.asList();
        final DataSetIterator trainIterator = new ListDataSetIterator<>(trainList,batchSize);

		// Create dataset iterators
        final List<DataSet> testList = allDataTest.asList();
        final DataSetIterator testIterator = new ListDataSetIterator<>(testList,batchSize);

        /*
            Construct the neural network
        */


        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		.seed(seed)
		.l2(0.0005)
		.weightInit(WeightInit.XAVIER)
		.updater(new Adam(1e-3))
		.list()
		//Using kernel size of (1,3) for every layer
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn defines number of channels
				.nIn(8)
				.stride(1,1)
				.nOut(32)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(32).build())
		.layer(new ConvolutionLayer.Builder(1, 3)
				.stride(1,1)
				.nOut(64)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(64).build())
		.layer(new ConvolutionLayer.Builder(1, 3)
				.stride(1,1)
				.nOut(64)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(64).build())
		.layer(new ConvolutionLayer.Builder(1, 3)
				.stride(1,1)
				.nOut(256)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		.layer(new ConvolutionLayer.Builder(1, 3)
				.stride(1,1)
				.nOut(256)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		//Classification network
		.layer(new ConvolutionLayer.Builder(1, 3)
				.stride(2, 2)
				.nOut(512)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		.layer(new ConvolutionLayer.Builder(1, 3)
				.stride(1,1)
				.nOut(512)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(1,1)
				.nOut(1024)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		.layer(new ConvolutionLayer.Builder(1, 1)
				.stride(1,1)
				.nOut(2) // Number of output classes
				.activation(Activation.IDENTITY)
				.build())
		// Defining loss function		
		.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
			.name("output")
			.nOut(2)
			.dropOut(0.8)
			.activation(Activation.SOFTMAX)
			.build())
		.setInputType(InputType.convolutional(1, 23, 8))
		.build();

		// Starting UI server
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        uiServer.attach(statsStorage);


		// Defining and initializing model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();



		// Train & evaluate
        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(1), new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        
        model.setListeners(new StatsListener(statsStorage));

        model.fit(trainIterator, nEpochs);

		testIterator.reset();

        Evaluation eval = model.evaluate(testIterator);

        log.info(eval.stats());

		testIterator.reset();

        ROC r  = model.evaluateROC(testIterator);

		testIterator.reset();

		log.info("ROC-AUC score of: " + String.valueOf(r.calculateAUC() ));

        log.info("****************Example finished********************");
    }
}

