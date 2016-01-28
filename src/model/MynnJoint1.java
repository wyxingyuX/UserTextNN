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

public class MynnJoint1 {


	LookupLinearTanh xseedLLT1Tags;
	LookupLinearTanh xseedLLT2Tags;
	//LookupLinearTanh xseedLLT3Tags;

	LookupLinearTanh xseedLLT1Weibos;
	LookupLinearTanh xseedLLT2Weibos;
	//LookupLinearTanh xseedLLT3Weibos;
   
	MultiConnectLayer connectTags;
	MultiConnectLayer connectWeibos;

	AverageLayer averageTags;
	AverageLayer averageWeibos;

	//LookupLayer userLookup;
	LookupLayer itemLookupTags;
	LookupLayer itemLookupWeibos;
	
	MultiConnectLayer connectItem;

	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;

	HashMap<String, Integer> wordTagsVocab = null;
	HashMap<String, Integer> wordWeibosVocab = null;

	HashMap<String, Integer> userVocab;
	HashMap<String, Integer> itemVocabTags;
	HashMap<String, Integer> itemVocabWeibos;

	String unkStr = "</s>";

	public MynnJoint1(
			String embeddingTagsFileWord, 
			int embeddingTagsLengthWord,
			String embeddingWeibosFileWord,
			int embeddingWeibosLengthWord,
			int windowSizeWordLookup1,
			int windowSizeWordLookup2,
			int windowSizeWordLookup3,
			int outputLengthWordLookup,
			int embeddingLengthItemLookup,
			int classNum,
			String trainTagsFile,
			String testTagsFile,
			String trainWeibosFile,
			String testWeibosFile,
			double randomizeBase) throws Exception
	{
		loadData(trainTagsFile, testTagsFile,trainWeibosFile,testWeibosFile);

		wordTagsVocab = new HashMap<String, Integer>();
		wordWeibosVocab=new  HashMap<String, Integer>();


		int embeddingTagsLineCount = Funcs.lineCounter(embeddingTagsFileWord, "utf8")-1;
		double[][] tableTags = new double[embeddingTagsLineCount][];
		Funcs.loadEmbeddingFile(embeddingTagsFileWord, embeddingTagsLengthWord, "utf8", 
				false, wordTagsVocab, tableTags);

		int embeddingWeibosLineCount = Funcs.lineCounter(embeddingWeibosFileWord, "utf8")-1;
		double[][] tableWeibos = new double[embeddingWeibosLineCount][];
		Funcs.loadEmbeddingFile(embeddingWeibosFileWord, embeddingWeibosLengthWord, "utf8", 
				false, wordWeibosVocab, tableWeibos);
		//////////////////////////////////////////
		xseedLLT1Tags = new LookupLinearTanh(windowSizeWordLookup1, wordTagsVocab.size(), outputLengthWordLookup, embeddingTagsLengthWord);
		xseedLLT1Tags.lookup.setEmbeddings(tableTags);

		xseedLLT2Tags = new LookupLinearTanh(windowSizeWordLookup2, wordTagsVocab.size(), outputLengthWordLookup, embeddingTagsLengthWord);
		xseedLLT2Tags.lookup.setEmbeddings(tableTags);

//		xseedLLT3Tags = new LookupLinearTanh(windowSizeWordLookup3, wordTagsVocab.size(), outputLengthWordLookup, embeddingTagsLengthWord);
//		xseedLLT3Tags.lookup.setEmbeddings(tableTags);

//		connectTags = new MultiConnectLayer(
//				new int[]{outputLengthWordLookup, outputLengthWordLookup, outputLengthWordLookup});
		connectTags = new MultiConnectLayer(
				new int[]{outputLengthWordLookup, outputLengthWordLookup});


		averageTags = new AverageLayer(connectTags.outputLength, outputLengthWordLookup);
		connectTags.link(averageTags);
		/////////
		xseedLLT1Weibos = new LookupLinearTanh(windowSizeWordLookup1, wordWeibosVocab.size(), outputLengthWordLookup, embeddingWeibosLengthWord);
		xseedLLT1Weibos.lookup.setEmbeddings(tableWeibos);

		xseedLLT2Weibos = new LookupLinearTanh(windowSizeWordLookup2, wordWeibosVocab.size(), outputLengthWordLookup, embeddingWeibosLengthWord);
		xseedLLT2Weibos.lookup.setEmbeddings(tableWeibos);

//		xseedLLT3Weibos = new LookupLinearTanh(windowSizeWordLookup3, wordWeibosVocab.size(), outputLengthWordLookup, embeddingWeibosLengthWord);
//		xseedLLT3Weibos.lookup.setEmbeddings(tableWeibos);
//
//		connectWeibos = new MultiConnectLayer(
//				new int[]{outputLengthWordLookup, outputLengthWordLookup, outputLengthWordLookup});
		connectWeibos = new MultiConnectLayer(
				new int[]{outputLengthWordLookup, outputLengthWordLookup});

		averageWeibos = new AverageLayer(connectWeibos.outputLength, outputLengthWordLookup);

		connectWeibos.link(averageWeibos);

		// user item lookup layers
		//userLookup = new LookupLayer(embeddingLengthUserLookup, userVocab.size(), 1);
		itemLookupTags = new LookupLayer(embeddingLengthItemLookup, itemVocabTags.size(), 1);
		itemLookupWeibos=new LookupLayer(embeddingLengthItemLookup, itemVocabWeibos.size(), 1);

		//		connectItem = new MultiConnectLayer(
		//				new int[]{averageTags.outputLength, userLookup.output.length, itemLookupTags.output.length});
		connectItem = new MultiConnectLayer(
				new int[]{averageTags.outputLength, averageWeibos.outputLength,itemLookupTags.output.length,itemLookupWeibos.output.length});

		averageTags.link(connectItem, 0);
		averageWeibos.link(connectItem, 1);
		//		userLookup.link(connectItem, 1);
		itemLookupTags.link(connectItem, 2);
		itemLookupWeibos.link(connectItem, 3);

		// linear for softmax
		linearForSoftmax = new LinearLayer(connectItem.outputLength, classNum);
		connectItem.link(linearForSoftmax);

		softmax = new SoftmaxLayer(classNum);
		linearForSoftmax.link(softmax);

		Random rnd = new Random(); 
		xseedLLT1Tags.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		xseedLLT2Tags.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		//xseedLLT3Tags.randomize(rnd, -1.0 * randomizeBase, randomizeBase);

		xseedLLT1Weibos.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		xseedLLT2Weibos.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		//xseedLLT3Weibos.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
	}

	List<Data> trainTagsDataList;
	List<Data> testTagsDataList; 
	List<Data> trainWeibosDataList;
	List<Data> testWeibosDataList;  

	public void loadData(
			String trainTagsFile,
			String testTagsFile,
			String trainWeibosFile,
			String testWeibosFile )
	{
		System.out.println("================ start loading corpus ==============");
		trainTagsDataList = new ArrayList<Data>();  
		trainWeibosDataList =new ArrayList<Data>(); 

		userVocab = new HashMap<String, Integer>();
		itemVocabTags = new HashMap<String, Integer>();
		itemVocabWeibos=new HashMap<String, Integer>();

		Funcs.loadCorpus(trainTagsFile, "utf8", trainTagsDataList);
		Funcs.loadCorpus(trainWeibosFile, "utf8", trainWeibosDataList);

		for(Data data: trainTagsDataList)
		{
			if(!userVocab.containsKey(data.userStr))
			{
				userVocab.put(data.userStr, userVocab.size());
			}
			if(!itemVocabTags.containsKey(data.productStr))
			{
				itemVocabTags.put(data.productStr, itemVocabTags.size());
			}
		}
		for(Data data: trainWeibosDataList)
		{
			if(!itemVocabWeibos.containsKey(data.productStr))
			{
				itemVocabWeibos.put(data.productStr, itemVocabWeibos.size());
			}
		}

		testTagsDataList = new ArrayList<Data>();
		testWeibosDataList=new ArrayList<Data>();
		Funcs.loadCorpus(testTagsFile, "utf8", testTagsDataList);
		Funcs.loadCorpus(testWeibosFile, "utf8", testWeibosDataList);

		System.out.println("training size: " + trainTagsDataList.size());
		System.out.println("testTagsDataList size: " + testTagsDataList.size());
		System.out.println("userVocab.size(): " + userVocab.size());
		System.out.println("itemVocabTags.size(): " + itemVocabTags.size());
		System.out.println("itemVocabWeibos.size(): " + itemVocabWeibos.size());
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
			//Collections.shuffle(trainTagsDataList, new Random());

			for(int idxData = 0; idxData < trainTagsDataList.size(); idxData++)
			{
				//////////////////////////
				Data dataWeibos = trainWeibosDataList.get(idxData);
				String[] sentencesWeibos = dataWeibos.reviewText.split("¡£{1,}|£¡{1,}|£¿{1,}|\\n");
				int[][] wordIdMatrixWeibos = Funcs.fillDocument(sentencesWeibos, wordWeibosVocab, unkStr);

				DocAverage docAverage1Weibos = new DocAverage(
						xseedLLT1Weibos,
						wordIdMatrixWeibos, 
						wordWeibosVocab.get(unkStr));

				DocAverage docAverage2Weibos= new DocAverage(
						xseedLLT2Weibos,
						wordIdMatrixWeibos, 
						wordWeibosVocab.get(unkStr));

//				DocAverage docAverage3Weibos = new DocAverage(
//						xseedLLT3Weibos,
//						wordIdMatrixWeibos, 
//						wordWeibosVocab.get(unkStr));

				if(docAverage1Weibos.sentenceConvList.size() == 0 
						|| docAverage2Weibos.sentenceConvList.size() == 0
						/*|| docAverage3Weibos.sentenceConvList.size() == 0*/)
				{
					System.out.println(dataWeibos.toString() + "docAverage.sentenceConvList.size() == 0");
					continue;
				}
			
				//////
				Data dataTags = trainTagsDataList.get(idxData);
				String[] sentencesTags = dataTags.reviewText.split("¡£{1,}|£¡{1,}|£¿{1,}|\\n");
				int[][] wordIdMatrixTags = Funcs.fillDocument(sentencesTags, wordTagsVocab, unkStr);

				DocAverage docAverage1Tags = new DocAverage(
						xseedLLT1Tags,
						wordIdMatrixTags, 
						wordTagsVocab.get(unkStr));

				DocAverage docAverage2Tags = new DocAverage(
						xseedLLT2Tags,
						wordIdMatrixTags, 
						wordTagsVocab.get(unkStr));

//				DocAverage docAverage3Tags = new DocAverage(
//						xseedLLT3Tags,
//						wordIdMatrixTags, 
//						wordTagsVocab.get(unkStr));

				if(docAverage1Tags.sentenceConvList.size() == 0 
						|| docAverage2Tags.sentenceConvList.size() == 0
						/*|| docAverage3Tags.sentenceConvList.size() == 0*/)
				{
					System.out.println(dataTags.toString() + "docAverage.sentenceConvList.size() == 0");
					continue;
				}

				////////////////////////////////////////////
				
				//userLookup.input[0] = userVocab.get(data.userStr);
				itemLookupTags.input[0] = itemVocabTags.get(dataTags.productStr);
				itemLookupWeibos.input[0]=itemVocabWeibos.get(dataWeibos.productStr);
				// important
				docAverage1Tags.link(connectTags, 0);
				docAverage2Tags.link(connectTags, 1);
				//docAverage3Tags.link(connectTags, 2);
				///////
				docAverage1Weibos.link(connectWeibos, 0);
				docAverage2Weibos.link(connectWeibos, 1);
				//docAverage3Weibos.link(connectWeibos, 2);


				// forward
				docAverage1Tags.forward();
				docAverage2Tags.forward();
				//docAverage3Tags.forward();
				connectTags.forward();
				averageTags.forward();
				////
				docAverage1Weibos.forward();
				docAverage2Weibos.forward();
				//docAverage3Weibos.forward();
				connectWeibos.forward();
				averageWeibos.forward();



				//userLookup.forward();
				itemLookupTags.forward();
				itemLookupWeibos.forward();
				connectItem.forward();

				linearForSoftmax.forward();
				softmax.forward();

				// set cross-entropy error 
				// we minus 1 because the saved goldRating is in range 1~5, while what we need is in range 0~4
				int goldRating = dataTags.goldRating - 1;
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
				itemLookupTags.backward();
				itemLookupWeibos.backward();

				averageTags.backward();
				connectTags.backward();
				docAverage1Tags.backward();
				docAverage2Tags.backward();
				//docAverage3Tags.backward();
				////
				averageWeibos.backward();
				connectWeibos.backward();
				docAverage1Weibos.backward();
				docAverage2Weibos.backward();
				//docAverage3Weibos.backward();

				// update
				linearForSoftmax.update(learningRate);

				docAverage1Tags.update(learningRate);
				docAverage2Tags.update(learningRate);
				//docAverage3Tags.update(learningRate);

				docAverage1Weibos.update(learningRate);
				docAverage2Weibos.update(learningRate);
				//docAverage3Weibos.update(learningRate);
				//userLookup.update(learningRate);
				itemLookupTags.update(learningRate);
				itemLookupWeibos.update(learningRate);

				// regularization ?
				//				if(lossC % regularizationFreq == 0)
				//				{
				//					docAverage.regularization(lambda);
				//				}
				// clearGrad
				docAverage1Tags.clearGrad();
				docAverage2Tags.clearGrad();
				//docAverage3Tags.clearGrad();
				connectTags.clearGrad();
				averageTags.clearGrad();
				/////
				docAverage1Weibos.clearGrad();
				docAverage2Weibos.clearGrad();
				//docAverage3Weibos.clearGrad();
				connectWeibos.clearGrad();
				averageWeibos.clearGrad();

				connectItem.clearGrad();
				//userLookup.clearGrad();
				itemLookupTags.clearGrad();
				itemLookupWeibos.clearGrad();

				linearForSoftmax.clearGrad();
				softmax.clearGrad();

				if(idxData % 100 == 0)
				{
					System.out.println("running idxData = " + idxData + "/" + trainTagsDataList.size() + "\t "
							+ "lossV/lossC = " + lossV + "/" + lossC + "\t"
							+ " = " + lossV/lossC
							+ "\t" + new Date().toLocaleString());
				}
			}

			//Funcs.dumpEmbedFile(dumpUserEmbeddingFile + "-" + round, "utf8", userVocab, userLookup.tableTags, userLookup.embeddingLength);
			Funcs.dumpEmbedFile(dumpItemEmbeddingFile+"\\tags.txt" + "-" + round, "utf8", itemVocabTags, itemLookupTags.table, itemLookupTags.embeddingLength);
			Funcs.dumpEmbedFile(dumpItemEmbeddingFile+"\\weibos.txt" + "-" + round, "utf8", itemVocabTags, itemLookupWeibos.table, itemLookupWeibos.embeddingLength);
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

		for(int idxData = 0; idxData < testTagsDataList.size(); idxData++)
		{
			Data dataWeibos = testWeibosDataList.get(idxData);
			String[] sentencesWeibos = dataWeibos.reviewText.split("¡£{1,}|£¡{1,}|£¿{1,}|\\n");
			int[][] wordIdMatrixWeibos = Funcs.fillDocument(sentencesWeibos, wordWeibosVocab, unkStr);

			DocAverage docAverage1Weibos = new DocAverage(
					xseedLLT1Weibos,
					wordIdMatrixWeibos, 
					wordWeibosVocab.get(unkStr));

			DocAverage docAverage2Weibos= new DocAverage(
					xseedLLT2Weibos,
					wordIdMatrixWeibos, 
					wordWeibosVocab.get(unkStr));

//			DocAverage docAverage3Weibos = new DocAverage(
//					xseedLLT3Weibos,
//					wordIdMatrixWeibos, 
//					wordWeibosVocab.get(unkStr));

			if(docAverage1Weibos.sentenceConvList.size() == 0 
					|| docAverage2Weibos.sentenceConvList.size() == 0
					/*|| docAverage3Weibos.sentenceConvList.size() == 0*/)
			{
				System.out.println(dataWeibos.toString() + "docAverage.sentenceConvList.size() == 0");
				continue;
			}
			////////////////////////////////////////////
			Data dataTags = testTagsDataList.get(idxData);
			String[] sentencesTags = dataTags.reviewText.split("¡£{1,}|£¡{1,}|£¿{1,}|\\n");
			int[][] wordIdMatrixTags = Funcs.fillDocument(sentencesTags, wordTagsVocab, unkStr);

			DocAverage docAverage1Tags = new DocAverage(
					xseedLLT1Tags,
					wordIdMatrixTags, 
					wordTagsVocab.get(unkStr));

			DocAverage docAverage2Tags = new DocAverage(
					xseedLLT2Tags,
					wordIdMatrixTags, 
					wordTagsVocab.get(unkStr));

//			DocAverage docAverage3Tags = new DocAverage(
//					xseedLLT3Tags,
//					wordIdMatrixTags, 
//					wordTagsVocab.get(unkStr));

			if(docAverage1Tags.sentenceConvList.size() == 0 
					|| docAverage2Tags.sentenceConvList.size() == 0
					/*|| docAverage3Tags.sentenceConvList.size() == 0*/)
			{
				System.out.println(dataTags.toString() + "docAverage.sentenceConvList.size() == 0");
				continue;
			}

			/////////////////////////////////////////////////

			//userLookup.input[0] = userVocab.get(data.userStr);
			itemLookupTags.input[0] = itemVocabTags.get(dataTags.productStr);
			itemLookupWeibos.input[0]=itemVocabWeibos.get(dataWeibos.productStr);
			
			// important
			docAverage1Tags.link(connectTags, 0);
			docAverage2Tags.link(connectTags, 1);
			//docAverage3Tags.link(connectTags, 2);
			///////
			docAverage1Weibos.link(connectWeibos, 0);
			docAverage2Weibos.link(connectWeibos, 1);
			//docAverage3Weibos.link(connectWeibos, 2);


			// forward
			docAverage1Tags.forward();
			docAverage2Tags.forward();
			//docAverage3Tags.forward();
			connectTags.forward();
			averageTags.forward();
			////
			docAverage1Weibos.forward();
			docAverage2Weibos.forward();
			//docAverage3Weibos.forward();
			connectWeibos.forward();
			averageWeibos.forward();


			//userLookup.forward();
			itemLookupTags.forward();
			itemLookupWeibos.forward();
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
			goldList.add(dataTags.goldRating);
		}

		Metric.calcMetric(goldList, predList);
		System.out.println("============== finish predicting =================");
	}

	public static void main(String[] args) throws Exception{
		//		String base="F:\\ExpData\\DataIntegate\\source\\nne\\Tags\\dyt\\";

		String base="F:\\ExpData\\DataIntegate\\source\\nne\\Joint\\";

		int embeddingLengthTags = 100;
		String embeddingFileTags =base+"Tags\\dyt\\wordVec\\word_vector.txt";
		int embeddingLengthWeibos = 100;
		String embeddingFileWeibos =base+"Weibos\\dyt\\wordVec\\word_vector.txt";
		//String embeddingFile =base+"wordVec\\word_vector.txt";
		// for yelp14 dataset, windowsize = 1&2&3 works best than other settings. 
		int windowSizeWordLookup1 = 1;
		int windowSizeWordLookup2 = 2;
		int windowSizeWordLookup3 = 3;
		int outputLengthWordLookup = 50;
		int classNum = 2;

		//int embeddingLengthUserLookup = 75;
		int embeddingLengthItemLookup = 50;

		String year = "amazon";
		String inputDirTags = base+"Tags\\dyt\\textData\\";
		String inputDirWeibos = base+"Weibos\\dyt\\textData\\";
		int roundNum = 20;
		double probThreshold = 0.001;
		double learningRate = 0.03;
		double randomizeBase = 0.01;

		String trainFileTags = inputDirTags + "\\yelp-" + year + "-train.txt.ss";
		String testFileTags = inputDirTags + "\\yelp-" + year + "-test.txt.ss";
		String trainFileWeibos = inputDirWeibos + "\\yelp-" + year + "-train.txt.ss";
		String testFileWeibos = inputDirWeibos + "\\yelp-" + year + "-test.txt.ss";

		if(year.equals("amazon"))
		{
			trainFileTags = inputDirTags + "/dytText_train.txt";
			testFileTags = inputDirTags + "/dytText_test.txt";
			
			trainFileWeibos = inputDirWeibos + "/dytText_train.txt";
			testFileWeibos = inputDirWeibos + "/dytText_test.txt";
		}

		//String dumpUserEmbeddingFile = base+"user-embedding\\2013-doc123-rerun-75dms.txt";
		String dumpItemEmbeddingFile = base+"source-embedding\\2013-doc123-rerun-50dms";
		//FileTool.guaranteeFileDirExist(dumpUserEmbeddingFile);
		FileTool.guaranteeFileDirExist(dumpItemEmbeddingFile);

		TestTool.println("Start...");
		MynnJoint1 main = new MynnJoint1(
				embeddingFileTags, 
				embeddingLengthTags, 
				embeddingFileWeibos, 
				embeddingLengthWeibos, 
				windowSizeWordLookup1,
				windowSizeWordLookup2,
				windowSizeWordLookup3,
				outputLengthWordLookup,
				embeddingLengthItemLookup,
				classNum, 
				trainFileTags, 
				testFileTags,
				trainFileWeibos,
				testFileWeibos,
				randomizeBase);

		main.run(roundNum, 
				probThreshold, 
				learningRate, 
				classNum,
				dumpItemEmbeddingFile);
		TestTool.println("End");

	}
}
