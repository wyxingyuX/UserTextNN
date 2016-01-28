package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import adapter.FileTool;
import adapter.TestTool;
import nnet.AverageLayer;
import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.MultiConnectLayer;
import nnet.SoftmaxLayer;
import other.Data;
import other.Funcs;
import other.Metric;

public class Mynn {

	LookupLinearTanh xseedLLT1;
	LookupLinearTanh xseedLLT2;
	LookupLinearTanh xseedLLT3;
	
	MultiConnectLayer connect;
	AverageLayer average;
	
	//LookupLayer userLookup;
	LookupLayer itemLookup;
	MultiConnectLayer connectItem;
	
	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;
	
	HashMap<String, Integer> wordVocab = null;
	
	HashMap<String, Integer> userVocab;
	HashMap<String, Integer> itemVocab;
	
	String unkStr = "</s>";
	
	public Mynn(
				String embeddingFileWord, 
				int embeddingLengthWord,
				int windowSizeWordLookup1,
				int windowSizeWordLookup2,
				int windowSizeWordLookup3,
				int outputLengthWordLookup,
				int embeddingLengthItemLookup,
				int classNum,
				String trainFile,
				String testFile,
				double randomizeBase) throws Exception
	{
		loadData(trainFile, testFile);
		
		wordVocab = new HashMap<String, Integer>();
		
		int embeddingLineCount = Funcs.lineCounter(embeddingFileWord, "utf8")-1;
		double[][] table = new double[embeddingLineCount][];
		Funcs.loadEmbeddingFile(embeddingFileWord, embeddingLengthWord, "utf8", 
				false, wordVocab, table);
		//////////////////////////////////////////
		xseedLLT1 = new LookupLinearTanh(windowSizeWordLookup1, wordVocab.size(), outputLengthWordLookup, embeddingLengthWord);
		xseedLLT1.lookup.setEmbeddings(table);
			
		xseedLLT2 = new LookupLinearTanh(windowSizeWordLookup2, wordVocab.size(), outputLengthWordLookup, embeddingLengthWord);
		xseedLLT2.lookup.setEmbeddings(table);
		
		xseedLLT3 = new LookupLinearTanh(windowSizeWordLookup3, wordVocab.size(), outputLengthWordLookup, embeddingLengthWord);
		xseedLLT3.lookup.setEmbeddings(table);
		
		connect = new MultiConnectLayer(
				new int[]{outputLengthWordLookup, outputLengthWordLookup, outputLengthWordLookup});
		
		average = new AverageLayer(connect.outputLength, outputLengthWordLookup);
		connect.link(average);
		
		// user item lookup layers
		//userLookup = new LookupLayer(embeddingLengthUserLookup, userVocab.size(), 1);
		itemLookup = new LookupLayer(embeddingLengthItemLookup, itemVocab.size(), 1);
		
//		connectItem = new MultiConnectLayer(
//				new int[]{average.outputLength, userLookup.output.length, itemLookup.output.length});
		connectItem = new MultiConnectLayer(
				new int[]{average.outputLength, itemLookup.output.length});
		
		average.link(connectItem, 0);
//		userLookup.link(connectItem, 1);
		itemLookup.link(connectItem, 1);
		
		// linear for softmax
		linearForSoftmax = new LinearLayer(connectItem.outputLength, classNum);
		connectItem.link(linearForSoftmax);
		
		softmax = new SoftmaxLayer(classNum);
		linearForSoftmax.link(softmax);
		
		Random rnd = new Random(); 
		xseedLLT1.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		xseedLLT2.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		xseedLLT3.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
	}
	
	List<Data> trainDataList;
	List<Data> testDataList;  
	
	public void loadData(
			String trainFile,
			String testFile)
	{
		System.out.println("================ start loading corpus ==============");
		trainDataList = new ArrayList<Data>();  
		
		userVocab = new HashMap<String, Integer>();
		itemVocab = new HashMap<String, Integer>();
		
		Funcs.loadCorpus(trainFile, "utf8", trainDataList);
		
		for(Data data: trainDataList)
		{
			if(!userVocab.containsKey(data.userStr))
			{
				userVocab.put(data.userStr, userVocab.size());
			}
			if(!itemVocab.containsKey(data.productStr))
			{
				itemVocab.put(data.productStr, itemVocab.size());
			}
		}
		
		testDataList = new ArrayList<Data>();  
		Funcs.loadCorpus(testFile, "utf8", testDataList);
		
		System.out.println("training size: " + trainDataList.size());
		System.out.println("testDataList size: " + testDataList.size());
		System.out.println("userVocab.size(): " + userVocab.size());
		System.out.println("itemVocab.size(): " + itemVocab.size());
		System.out.println("================ finsh loading corpus ==============");
	}
	
	public void run(
			int roundNum,
			double probThreshould,
			double learningRate,
			int classNum,
			String dumpItemEmbeddingFile
			) throws Exception
	{
		double lossV = 0.0;
		int lossC = 0;
		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");
			//Collections.shuffle(trainDataList, new Random());

			for(int idxData = 0; idxData < trainDataList.size(); idxData++)
			{
				Data data = trainDataList.get(idxData);
				
				//String[] sentences = data.reviewText.split("<sssss>");
				String[] sentences = data.reviewText.split("¡£{1,}|£¡{1,}|£¿{1,}|\\n");
				int[][] wordIdMatrix = Funcs.fillDocument(sentences, wordVocab, unkStr);
				
				DocAverage docAverage1 = new DocAverage(
						xseedLLT1,
						wordIdMatrix, 
						wordVocab.get(unkStr));
				
				DocAverage docAverage2 = new DocAverage(
						xseedLLT2,
						wordIdMatrix, 
						wordVocab.get(unkStr));
				
				DocAverage docAverage3 = new DocAverage(
						xseedLLT3,
						wordIdMatrix, 
						wordVocab.get(unkStr));
				
				if(docAverage1.sentenceConvList.size() == 0 
						|| docAverage2.sentenceConvList.size() == 0
						|| docAverage3.sentenceConvList.size() == 0)
				{
					System.out.println(data.toString() + "docAverage.sentenceConvList.size() == 0");
					continue;
				}
				
				//userLookup.input[0] = userVocab.get(data.userStr);
				itemLookup.input[0] = itemVocab.get(data.productStr);
				
				// important
				docAverage1.link(connect, 0);
				docAverage2.link(connect, 1);
				docAverage3.link(connect, 2);
				
				// forward
 				docAverage1.forward();
 				docAverage2.forward();
 				docAverage3.forward();
 				connect.forward();
				average.forward();

				//userLookup.forward();
				itemLookup.forward();
				connectItem.forward();
				
 				linearForSoftmax.forward();
				softmax.forward();
				
				// set cross-entropy error 
				// we minus 1 because the saved goldRating is in range 1~5, while what we need is in range 0~4
				int goldRating = data.goldRating - 1;
				lossV += -Math.log(softmax.output[goldRating]);
				lossC += 1;
				
				for(int k = 0; k < softmax.outputG.length; k++)
					softmax.outputG[k] = 0.0;
				
				if(softmax.output[goldRating] < probThreshould)
					softmax.outputG[goldRating] =  1.0 / probThreshould;
				else
					softmax.outputG[goldRating] = 1.0 / softmax.output[goldRating];
				
				// backward
				softmax.backward();
				linearForSoftmax.backward();
				
				connectItem.backward();
				//userLookup.backward();
				itemLookup.backward();
				
				average.backward();
				connect.backward();
				docAverage1.backward();
				docAverage2.backward();
				docAverage3.backward();
				
				// update
				linearForSoftmax.update(learningRate);
				docAverage1.update(learningRate);
				docAverage2.update(learningRate);
				docAverage3.update(learningRate);
				//userLookup.update(learningRate);
				itemLookup.update(learningRate);
				
				// regularization ?
//				if(lossC % regularizationFreq == 0)
//				{
//					docAverage.regularization(lambda);
//				}
				// clearGrad
				docAverage1.clearGrad();
				docAverage2.clearGrad();
				docAverage3.clearGrad();
				connect.clearGrad();
				average.clearGrad();

				connectItem.clearGrad();
				//userLookup.clearGrad();
				itemLookup.clearGrad();
				
				linearForSoftmax.clearGrad();
				softmax.clearGrad();
				
				if(idxData % 100 == 0)
				{
					System.out.println("running idxData = " + idxData + "/" + trainDataList.size() + "\t "
							+ "lossV/lossC = " + lossV + "/" + lossC + "\t"
							+ " = " + lossV/lossC
							+ "\t" + new Date().toLocaleString());
				}
			}
			
			//Funcs.dumpEmbedFile(dumpUserEmbeddingFile + "-" + round, "utf8", userVocab, userLookup.table, userLookup.embeddingLength);
			Funcs.dumpEmbedFile(dumpItemEmbeddingFile + "-" + round, "utf8", itemVocab, itemLookup.table, itemLookup.embeddingLength);
			
			System.out.println("============= finish training round: " + round + " ==============");
			
			predict(round);
		}
		//predict(roundNum);
	}
	
	public void predict(int round) throws Exception
	{
		System.out.println("=========== predicting round: " + round + " ===============");
		
		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();
		
		for(int idxData = 0; idxData < testDataList.size(); idxData++)
		{
			Data data = testDataList.get(idxData);
			
			//String[] sentences = data.reviewText.split("<sssss>");
			String[] sentences = data.reviewText.split("¡£{1,}|£¡{1,}|£¿{1,}|\\n");
			int[][] wordIdMatrix = Funcs.fillDocument(sentences, wordVocab, unkStr);
			
			DocAverage docAverage1 = new DocAverage(
					xseedLLT1,
					wordIdMatrix, 
					wordVocab.get(unkStr));
			
			DocAverage docAverage2 = new DocAverage(
					xseedLLT2,
					wordIdMatrix, 
					wordVocab.get(unkStr));
			
			DocAverage docAverage3 = new DocAverage(
					xseedLLT3,
					wordIdMatrix, 
					wordVocab.get(unkStr));
			
			if(docAverage1.sentenceConvList.size() == 0 
					|| docAverage2.sentenceConvList.size() == 0
					|| docAverage3.sentenceConvList.size() == 0)
			{
				System.out.println(data.toString() + "docAverage.sentenceConvList.size() == 0");
				continue;
			}
			
			//userLookup.input[0] = userVocab.get(data.userStr);
			itemLookup.input[0] = itemVocab.get(data.productStr);
			
			// important
			docAverage1.link(connect, 0);
			docAverage2.link(connect, 1);
			docAverage3.link(connect, 2);
			
			// forward
			docAverage1.forward();
			docAverage2.forward();
			docAverage3.forward();
			connect.forward();
			average.forward();

			//userLookup.forward();
			itemLookup.forward();
			connectItem.forward();
			
			linearForSoftmax.forward();
			softmax.forward();
			
			int predClass = -1;
			double maxPredProb = -1.0;
			for(int ii = 0; ii < softmax.length; ii++)
			{
				if(softmax.output[ii] > maxPredProb)
				{
					maxPredProb = softmax.output[ii];
					predClass = ii;
				}
			}
			
			predList.add(predClass + 1);
			goldList.add(data.goldRating);
		}
		
		Metric.calcMetric(goldList, predList);
		System.out.println("============== finish predicting =================");
	}
	
	public static void main(String[] args) throws Exception{
	   //String base="F:\\ExpData\\DataIntegate\\source\\nne\\Tags\\dyt\\";
		
		String base="F:\\ExpData\\DataIntegate\\source\\nne\\Weibos\\extract5Weibo\\dyt\\";
		
		int embeddingLength = 100;
		//String embeddingFile =base+"wordVec\\weibosFc_vector.txt";
		String embeddingFile =base+"wordVec\\word_vector.txt";
		// for yelp14 dataset, windowsize = 1&2&3 works best than other settings. 
		int windowSizeWordLookup1 = 1;
		int windowSizeWordLookup2 = 2;
		int windowSizeWordLookup3 = 3;
		int outputLengthWordLookup = 50;
		int classNum = 2;
		
		//int embeddingLengthUserLookup = 75;
		int embeddingLengthItemLookup = 50;
		
		String year = "amazon";
		String inputDir = base+"textData\\";
		int roundNum = 110;
		double probThreshold = 0.001;
		double learningRate = 0.03;
		double randomizeBase = 0.01;
		
		String trainFile = inputDir + "\\yelp-" + year + "-train.txt.ss";
		String testFile = inputDir + "\\yelp-" + year + "-test.txt.ss";
		
		if(year.equals("amazon"))
		{
			trainFile = inputDir + "/dytText_train.txt";
			testFile = inputDir + "/dytText_test.txt";
		}
		
		//String dumpUserEmbeddingFile = base+"user-embedding\\2013-doc123-rerun-75dms.txt";
		String dumpItemEmbeddingFile = base+"source-embedding\\2013-doc123-rerun-50dms.txt";
		//FileTool.guaranteeFileDirExist(dumpUserEmbeddingFile);
		FileTool.guaranteeFileDirExist(dumpItemEmbeddingFile);
		
		TestTool.println("Start...");
		Mynn main = new Mynn(
				embeddingFile, 
				embeddingLength, 
				windowSizeWordLookup1,
				windowSizeWordLookup2,
				windowSizeWordLookup3,
				outputLengthWordLookup,
				embeddingLengthItemLookup,
				classNum, 
				trainFile, 
				testFile,
				randomizeBase);
		
		main.run(roundNum, 
				probThreshold, 
				learningRate, 
				classNum,
				dumpItemEmbeddingFile);
		TestTool.println("End");
		
	}

}
