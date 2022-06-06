///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS org.deeplearning4j:deeplearning4j-core:1.0.0-M2
//DEPS org.nd4j:nd4j-native:1.0.0-M2
/* *****************************************************************************
 *
 * This is an implementation of DeepCrispr in DeepLearning4j
 * Data is read in through numpy files using the ND4J library
 * A t
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


import java.io.File;


public class deepCrispr {
    private static final Logger log = LoggerFactory.getLogger(deepCrispr.class);

    public static void main(String[] args) throws Exception {
        int batchSize = 64; // Batch size
        int nEpochs = 10; // Number of training epochs
        int seed = 123; //

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

        final List<DataSet> testList = allDataTest.asList();
        final DataSetIterator testIterator = new ListDataSetIterator<>(testList,batchSize);

        // DataSetIterator train_set_iterator  = new INDArrayDataSetIterator(train_X, train_y, batchSize);
        // DataSetIterator test_set_iterator  = new INDArrayDataSetIterator(test_X, test_y, batchSize);

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
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.nIn(8)
				.stride(1,1)
				.nOut(32)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(32).build())
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(1,1)
				.nOut(64)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(64).build())
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(1,1)
				.nOut(64)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(64).build())
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(1,1)
				.nOut(256)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(1,1)
				.nOut(256)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		//Decode network
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(2, 2)
				.nOut(512)
				.activation(Activation.IDENTITY)
				.build())
		.layer(new BatchNormalization.Builder().nOut(256).build())		
		.layer(new ConvolutionLayer.Builder(1, 3)
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
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
				//nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
				.stride(1,1)
				.nOut(2)
				.activation(Activation.IDENTITY)
				.build())
				
		.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
			.name("output")
			.nOut(2)
			.dropOut(0.8)
			.activation(Activation.SOFTMAX)
			.build())
		.setInputType(InputType.convolutional(1, 23, 8))
		.build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)

        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        uiServer.attach(statsStorage);



        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();




        log.info("Train model...");
        model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        
        model.setListeners(new StatsListener(statsStorage));

        model.fit(trainIterator, nEpochs);

        Evaluation eval = model.evaluate(testIterator);

        log.info(eval.stats());


        // int roc = model.evaluateROC(testIterator);


        log.info("****************Example finished********************");
    }
}

