package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import adapter.FileTool;
import adapter.TestTool;
import adapter.ToolKit;
import nnet.AverageLayer;
import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.MultiConnectLayer;
import nnet.SoftmaxLayer;
import other.Data;
import other.Funcs;
import other.Metric;
import wyConnect.CNN4ViewUser;
import wyConnect.View;

public class MynnForCL {
	List<CNN4ViewUser> net4UserList;

	LookupLayer fusionLookup;
	MultiConnectLayer connectItem;

	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;

	//	HashMap<String, Integer> userVocab;
	//HashMap<String, Integer> fusionVocab;
	public MynnForCL(
			List<View> views,
			List<Integer> windowSizeWordLookupList,
			int outputLengthWordLookup,
			int embeddingLengthItemLookup,
			int classNum,
			double randomizeBase) throws Exception{
		init(views,windowSizeWordLookupList,outputLengthWordLookup, embeddingLengthItemLookup,classNum,randomizeBase);
	}
	public void init(
			List<View> views,
			List<Integer> windowSizeWordLookupList,
			int outputLengthWordLookup,
			int embeddingLengthItemLookup,
			int classNum,
			double randomizeBase) throws Exception{
		net4UserList=new ArrayList<CNN4ViewUser>();
		for(View view:views){
			CNN4ViewUser net4User=new CNN4ViewUser(view,windowSizeWordLookupList,outputLengthWordLookup);
			net4UserList.add(net4User);
		}  

		//item lookup layers
		fusionLookup = new LookupLayer(embeddingLengthItemLookup,1, 1);

		int[] multConInputlengths=new int[net4UserList.size()+1];
		int i=0;
		for(;i<net4UserList.size();++i){
			multConInputlengths[i]=net4UserList.get(i).average.outputLength;
		}
		multConInputlengths[i]=fusionLookup.output.length;
		connectItem = new MultiConnectLayer(multConInputlengths);

		i=0;
		for(;i<net4UserList.size();++i){
			net4UserList.get(i).average.link(connectItem, i);
		}
		fusionLookup.link(connectItem, i);

		// linear for softmax
		linearForSoftmax = new LinearLayer(connectItem.outputLength, classNum);
		connectItem.link(linearForSoftmax);

		softmax = new SoftmaxLayer(classNum);
		linearForSoftmax.link(softmax);

		Random rnd = new Random(); 
		for(CNN4ViewUser net4User:net4UserList){
			net4User.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		}
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
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
		
		int lastRound=-1;
		
		for(CNN4ViewUser n4u:net4UserList){
			Collections.shuffle(n4u.trainDataList, new Random(1000));
		}
		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");
            
			//һ��Ҫ��֤ ��viewDataÿ����ͬһ���û�
			for(int idxData = 0; idxData < net4UserList.get(0).trainDataList.size(); idxData++)
			{
				Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap=new LinkedHashMap<CNN4ViewUser,List<DocAverage>>();
				for(CNN4ViewUser net4User:net4UserList){
					List<DocAverage> docAverageList=net4User.getDocAverageList(idxData,"train");
					viewUserDocAverageListMap.put(net4User, docAverageList);
				}

				if(!this.idDocAverageListValid(viewUserDocAverageListMap))
				{
					//System.out.println(data.toString() + "docAverage.sentenceConvList.size() == 0");
					continue;
				}

				//fusionLookup.input[0] = fusionVocab.get(data.productStr);//
				fusionLookup.input[0] =0;

				// important
				this.net4ViewUserListLinkAndForward(viewUserDocAverageListMap);

				//userLookup.forward();
				fusionLookup.forward();
				connectItem.forward();

				linearForSoftmax.forward();
				softmax.forward();

				// set cross-entropy error 
				// we minus 1 because the saved goldRating is in range 1~5, while what we need is in range 0~4
				Data data=net4UserList.get(0).trainDataList.get(idxData);
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
				fusionLookup.backward();

				this.net4ViewUserListBackward(viewUserDocAverageListMap);

				// update
				linearForSoftmax.update(learningRate);
				this.net4ViewUserListUpdate(viewUserDocAverageListMap, learningRate);
				//userLookup.update(learningRate);
				fusionLookup.update(learningRate);

				
				////////write docsAverage to file
//				boolean isAppend=true;
//				if(lastRound!=round){
//					isAppend=false;
//					lastRound=round;
//				}
//				for(CNN4ViewUser net4User:net4UserList){
//					ToolKit.writeDocsAverage2File(net4User, FileTool.getParentPath(net4User.trainFile)+"\\trainAverage\\average.txt",isAppend);
//				}
				/////////////
				// clearGrad
				this.net4ViewUserListClearGrad(viewUserDocAverageListMap);

				connectItem.clearGrad();
				fusionLookup.clearGrad();

				linearForSoftmax.clearGrad();
				softmax.clearGrad();

				if(idxData % 100 == 0)
				{
					System.out.println("running idxData = " + idxData + "/" + net4UserList.get(0).trainDataList.size() + "\t "
							+ "lossV/lossC = " + lossV + "/" + lossC + "\t"
							+ " = " + lossV/lossC
							+ "\t" + new Date().toLocaleString());
				}
			}

			//Funcs.dumpEmbedFile(dumpUserEmbeddingFile + "-" + round, "utf8", userVocab, userLookup.table, userLookup.embeddingLength);
			//Funcs.dumpEmbedFile(dumpItemEmbeddingFile + "-" + round, "utf8", fusionVocab, fusionLookup.table, fusionLookup.embeddingLength);

			System.out.println("============= finish training round: " + round + " ==============");

			predict(round,lastRound);
		}
		//predict(roundNum);
	}

	public void predict(int round,int lastRound) throws Exception
	{
		System.out.println("=========== predicting round: " + round + " ===============");

		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();
        
		for(int idxData = 0; idxData < net4UserList.get(0).testDataList.size(); idxData++)
		{
			Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap=new LinkedHashMap<CNN4ViewUser,List<DocAverage>>();
			for(CNN4ViewUser net4User:net4UserList){
				List<DocAverage> docAverageList=net4User.getDocAverageList(idxData,"test");
				viewUserDocAverageListMap.put(net4User, docAverageList);
			}

			if(!this.idDocAverageListValid(viewUserDocAverageListMap))
			{
				//System.out.println(data.toString() + "docAverage.sentenceConvList.size() == 0");
				continue;
			}

			//userLookup.input[0] = userVocab.get(data.userStr);
			fusionLookup.input[0] =0;

			// important
			this.net4ViewUserListLinkAndForward(viewUserDocAverageListMap);

			//userLookup.forward();
			fusionLookup.forward();
			connectItem.forward();

			linearForSoftmax.forward();
			softmax.forward();
            
			////////write docsAverage to file
			boolean isAppend=true;
			if(lastRound!=round){
				isAppend=false;
				lastRound=round;
			}
			for(CNN4ViewUser net4User:net4UserList){
				ToolKit.writeDocsAverage2File(net4User, FileTool.getParentPath(net4User.trainFile)+"\\trainAverage\\average.txt",isAppend);
			}
			/////////////
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
            Data data=net4UserList.get(0).testDataList.get(idxData);
			predList.add(predClass + 1);
			goldList.add(data.goldRating);
		}

		Metric.calcMetric(goldList, predList);
		System.out.println("============== finish predicting =================");
	}

	public void net4ViewUserLinkAndForward(List<DocAverage> docAverageList,CNN4ViewUser net4User ) throws Exception{
		//link
		int i=0;
		for(DocAverage docAverage: docAverageList){
			docAverage.link(net4User.connect, i);
			++i;
		}

		//forward
		for(DocAverage docAverage: docAverageList){
			docAverage.forward();
		}
		net4User.connect.forward();
		net4User.average.forward();
	}
	
	public void net4ViewUserListLinkAndForward(Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap) throws Exception{
		for(Map.Entry<CNN4ViewUser,List<DocAverage>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<DocAverage> docAverageList=entry.getValue();
			//link
			int i=0;
			for(DocAverage docAverage: docAverageList){
				docAverage.link(net4User.connect, i);
				++i;
			}

			//forward
			for(DocAverage docAverage: docAverageList){
				docAverage.forward();
			}
			net4User.connect.forward();
			net4User.average.forward();
		}
	}

	public void net4ViewUserListBackward(Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser,List<DocAverage>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<DocAverage> docAverageList=entry.getValue();

			net4User.average.backward();
			net4User.connect.backward();
			for(DocAverage docAverage: docAverageList){
				docAverage.backward();
			}
		}
	}

	public void net4ViewUserListUpdate(Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap,double learningRate){
		for(Map.Entry<CNN4ViewUser,List<DocAverage>> entry:viewUserDocAverageListMap.entrySet()){
			List<DocAverage> docAverageList=entry.getValue();

			for(DocAverage docAverage:docAverageList){
				docAverage.update(learningRate);
			}
		}
	}

	public void net4ViewUserListClearGrad(Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser,List<DocAverage>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<DocAverage> docAverageList=entry.getValue();

			for(DocAverage docAverage:docAverageList){
				docAverage.clearGrad();
			}
			net4User.connect.clearGrad();
			net4User.average.clearGrad();
		}
	}
	protected boolean idDocAverageListValid(Map<CNN4ViewUser,List<DocAverage>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser,List<DocAverage>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<DocAverage> docAverageList=entry.getValue();
			
			for(DocAverage docAverage:docAverageList){
				if(docAverage.sentenceConvList.size()==0){
					System.out.println(net4User.getData().toString()+" docAverage.sentenceConvList.size() == 0");
					return false;
				}
			}
		}
		return true;
	}

	public static void main(String[] args) throws Exception{
		String base="F:\\ExpData\\DataIntegate\\source\\nne\\JointTest\\";

		//String base="F:\\ExpData\\DataIntegate\\source\\nne\\Weibos\\dyt\\";

		int embeddingLength = 100;
		String embeddingFileWeibos =base+"Weibos\\dyt\\wordVec\\word_vector.txt";
		String trainFileWeibos =base + "Weibos\\dyt\\textData\\dytText_train.txt";
		String testFileWeibos =base + "Weibos\\dyt\\textData\\dytText_test.txt";
		
		String embeddingFileTags =base+"Tags\\dyt\\wordVec\\word_vector.txt";
		String trainFileTags =base + "Tags\\dyt\\textData\\dytText_train.txt";
		String testFileTags =base + "Tags\\dyt\\textData\\dytText_test.txt";
		// for yelp14 dataset, windowsize = 1&2&3 works best than other settings. 
		int outputLengthWordLookup = 50;
		int classNum = 2;

		//int embeddingLengthUserLookup = 75;
		int embeddingLengthItemLookup = 50;

		String inputDir = base+"textData\\";
		int roundNum = 100;
		double probThreshold = 0.001;
		double learningRate = 0.03;
		double randomizeBase = 0.01;


		//String dumpUserEmbeddingFile = base+"user-embedding\\2013-doc123-rerun-75dms.txt";
		//String dumpItemEmbeddingFile = base+"source-embedding\\2013-doc123-rerun-50dms.txt";
		//FileTool.guaranteeFileDirExist(dumpUserEmbeddingFile);
		//FileTool.guaranteeFileDirExist(dumpItemEmbeddingFile);
		
        List<View> views=new ArrayList<View>();
        //views.add(new View(embeddingFileWeibos,embeddingLength,trainFileWeibos,testFileWeibos));
        views.add(new View(embeddingFileTags,embeddingLength,trainFileTags,testFileTags));
        
        List<Integer> windowSizeWordLookupList=new ArrayList<Integer>();
        windowSizeWordLookupList.add(1);
        windowSizeWordLookupList.add(2);
        //windowSizeWordLookupList.add(3);
        
		TestTool.println("Start...");
		MynnForCL main = new MynnForCL(
				views,
				windowSizeWordLookupList,
				outputLengthWordLookup,
				embeddingLengthItemLookup,
				classNum, 
				randomizeBase);
		main.run(roundNum, 
				probThreshold, 
				learningRate, 
				classNum,
				"");
		TestTool.println("End");

	}
}
