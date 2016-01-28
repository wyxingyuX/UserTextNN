package model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import adapter.FileTool;
import adapter.GlobleLog;
import adapter.TestTool;
import adapter.ToolKit;
import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.MultiConnectLayer;
import nnet.NNInterface;
import nnet.SoftmaxLayer;
import other.Data;
import other.Metric;
import wyConnect.CNN4ViewUser1;
import wyConnect.LinearTanh;
import wyConnect.View;

public class Mynn7 {
	List<CNN4ViewUser1> net4UserList;
    
	
	LookupLayer fusionLookup;
	MultiConnectLayer connectFusion;

	LinearTanh linearTanh1;
	
	LinearTanh linearTanh2;

	SoftmaxLayer softmax;

	//	HashMap<String, Integer> userVocab;
	//HashMap<String, Integer> fusionVocab;
	public Mynn7(
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
		net4UserList=new ArrayList<CNN4ViewUser1>();
		for(View view:views){
			CNN4ViewUser1 net4User=new CNN4ViewUser1(view.embeddingFileWord,view.embeddingLengthWord,windowSizeWordLookupList,outputLengthWordLookup,view.trainIds,view.testIds,view.allDatas);
			net4UserList.add(net4User);
		}  

		//fusion lookup layers
		fusionLookup = new LookupLayer(embeddingLengthItemLookup,1, 1);

		int[] multConInputlengths=new int[net4UserList.size()+1];
		int i=0;
		for(;i<net4UserList.size();++i){
			multConInputlengths[i]=net4UserList.get(i).average.outputLength;
		}
		multConInputlengths[i]=fusionLookup.output.length;
		connectFusion = new MultiConnectLayer(multConInputlengths);

		i=0;
		for(;i<net4UserList.size();++i){
			net4UserList.get(i).average.link(connectFusion, i);
		}
		fusionLookup.link(connectFusion, i);

		// linearTanh for softmax
	    int hiddenNum=connectFusion.outputLength/net4UserList.size();
		linearTanh1 = new LinearTanh(connectFusion.outputLength, hiddenNum);
		connectFusion.link(linearTanh1);
		
		linearTanh2 = new LinearTanh(linearTanh1.outputLength, classNum);
		linearTanh1.link(linearTanh2);

		softmax = new SoftmaxLayer(linearTanh2.outputLength);
		linearTanh2.link(softmax);
		//linearTanh1.link(softmax);

		Random rnd = new Random(); 
		for(CNN4ViewUser1 net4User:net4UserList){
			net4User.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		}
		linearTanh1.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		linearTanh2.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
	}

	public double run(
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

		List<Double> accs=new ArrayList<Double>();
		for(CNN4ViewUser1 n4u:net4UserList){
			Collections.shuffle(n4u.trainDataList, new Random(1000));
		}
		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");

			for(int idxData = 0; idxData < net4UserList.get(0).trainDataList.size(); idxData++)
			{
				Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap=new LinkedHashMap<CNN4ViewUser1,List<NNInterface>>();
				for(CNN4ViewUser1 net4User:net4UserList){
					List<NNInterface> docAverageList=net4User.getDocAverageList(idxData,"train");
					viewUserDocAverageListMap.put(net4User, docAverageList);
				}

				if(!this.idDocAverageListValid(viewUserDocAverageListMap))
				{
					continue;
				}

				//fusionLookup.input[0] = fusionVocab.get(data.productStr);//
				fusionLookup.input[0] =0;

				// important
				this.net4ViewUserListLinkAndForward(viewUserDocAverageListMap);
				fusionLookup.forward();

				connectFusion.forward();

				linearTanh1.forward();
				linearTanh2.forward();
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
				linearTanh2.backward();
				linearTanh1.backward();

				connectFusion.backward();
				fusionLookup.backward();

				this.net4ViewUserListBackward(viewUserDocAverageListMap);

				// update
				linearTanh2.update(learningRate);
				linearTanh1.update(learningRate);
				this.net4ViewUserListUpdate(viewUserDocAverageListMap, learningRate);
				fusionLookup.update(learningRate);


				////////write docsAverage to file
				//				boolean isAppend=true;
				//				if(lastRound!=round){
				//					isAppend=false;
				//					lastRound=round;
				//				}
				////				for(Net4ViewUser1 net4User:net4UserList){
				////					ToolKit.writeDocsAverage2File(net4User, FileTool.getParentPath(net4User.trainFile)+"\\trainAverage\\average.txt",isAppend);
				////				}
				//				/////////////
				// clearGrad
				this.net4ViewUserListClearGrad(viewUserDocAverageListMap);

				connectFusion.clearGrad();
				fusionLookup.clearGrad();

				linearTanh1.clearGrad();
				linearTanh2.clearGrad();
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

			double acc=predict(round,lastRound);
			accs.add(acc);
		}
		//predict(roundNum);
		Map.Entry<Integer,Double> maxValueMap=ToolKit.getMaxValue(accs);
		TestTool.println("=============total "+roundNum+" epochs"+" =====================");
		TestTool.println("MaxAcc="+maxValueMap.getValue());
		TestTool.println("MaxAccRound="+maxValueMap.getKey());
		TestTool.println("========================================================== finnish in "+new Date()+"============================================================");
		return maxValueMap.getValue();
	}

	public double predict(int round,int lastRound) throws Exception
	{
		TestTool.println("=========== predicting round: " + round + " ===============");

		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();

		for(int idxData = 0; idxData < net4UserList.get(0).testDataList.size(); idxData++)
		{
			Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap=new LinkedHashMap<CNN4ViewUser1,List<NNInterface>>();
			for(CNN4ViewUser1 net4User:net4UserList){
				List<NNInterface> docAverageList=net4User.getDocAverageList(idxData,"test");
				viewUserDocAverageListMap.put(net4User, docAverageList);
			}

			if(!this.idDocAverageListValid(viewUserDocAverageListMap))
			{
				continue;
			}

			//userLookup.input[0] = userVocab.get(data.userStr);
			fusionLookup.input[0] =0;

			// important
			this.net4ViewUserListLinkAndForward(viewUserDocAverageListMap);

			fusionLookup.forward();
			connectFusion.forward();

			linearTanh1.forward();
			linearTanh2.forward();
			softmax.forward();

			////////write docsAverage to file
			//			boolean isAppend=true;
			//			if(lastRound!=round){
			//				isAppend=false;
			//				lastRound=round;
			//			}
			////			for(Net4ViewUser1 net4User:net4UserList){
			////				ToolKit.writeDocsAverage2File(net4User, FileTool.getParentPath(net4User.trainFile)+"\\trainAverage\\average.txt",isAppend);
			////			}
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

		double acc=Metric.calcMetric(goldList, predList);
		TestTool.println("============== finish predicting =================");
		return acc;
	}

	public void net4ViewUserListLinkAndForward(Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap) throws Exception{
		for(Map.Entry<CNN4ViewUser1,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser1 net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();
			//link
			int i=0;
			for(NNInterface docAverage: docAverageList){
				docAverage.link(net4User.connect, i);
				++i;
			}

			net4User.viewLookup.link(net4User.connect, i);

			//forward
			for(NNInterface docAverage: docAverageList){
				docAverage.forward();
			}
			net4User.viewLookup.forward();
			net4User.connect.forward();
			net4User.viewLinerLayer.forward();
			net4User.average.forward();

		}
	}

	public void net4ViewUserListBackward(Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser1,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser1 net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			net4User.average.backward();
			net4User.viewLinerLayer.backward();
			net4User.connect.backward();
			net4User.viewLookup.backward();
			for(NNInterface docAverage: docAverageList){
				docAverage.backward();
			}
		}
	}

	public void net4ViewUserListUpdate(Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap,double learningRate){
		for(Map.Entry<CNN4ViewUser1,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser1 net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();
            
			net4User.viewLookup.update(learningRate);
			net4User.viewLinerLayer.update(learningRate);
			for(NNInterface docAverage:docAverageList){
				docAverage.update(learningRate);
			}
		}
	}

	public void net4ViewUserListClearGrad(Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser1,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser1 net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			for(NNInterface docAverage:docAverageList){
				docAverage.clearGrad();
			}
			net4User.viewLookup.clearGrad();

			net4User.connect.clearGrad();
			net4User.viewLinerLayer.clearGrad();
			net4User.average.clearGrad();
		}
	}
	protected boolean idDocAverageListValid(Map<CNN4ViewUser1,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser1,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser1 net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			for(NNInterface docAverage:docAverageList){
				if(!docAverage.isValid()){
					System.out.println(net4User.getData().toString()+" docAverage.sentenceConvList.size() == 0");
					return false;
				}
			}
		}
		return true;
	}

	public static void main(String[] args) throws Exception{
		String sourceBase="F:\\ExpData\\DataIntegate\\source\\nne\\publicInfo\\";
		String destBase=FileTool.forwardReplaceDirNode(sourceBase, "source", "dest");
		int embeddingLength = 100;
		
		String view1Name="Weibos";
		String embeddingFileView1 =sourceBase+view1Name+"\\wordVec\\word_vector.txt";
		String allDataView1=sourceBase+view1Name+"\\textData\\allReview.txt";

		String view2Name="Tag-friTagVec";  
		String embeddingFileView2 =sourceBase+view2Name+"\\wordVec\\FriTag_vector.txt";
		String allDataView2=sourceBase+view2Name+"\\textData\\allReview.txt";
		
		String view3Name="Tag-wordVec";  
		String embeddingFileView3 =sourceBase+view2Name+"\\wordVec\\word_vector.txt";
		String allDataView3=sourceBase+view2Name+"\\textData\\allReview.txt";

		String destName="";

		List<Double> accs=new ArrayList<Double>();
		int totalFold=5;
		for(int fold=0;fold<totalFold;++fold){
			String trainIds = sourceBase+"5fold\\"+fold+"\\train.txt";
			String testIds =sourceBase+"5fold\\"+fold+"\\test.txt";

			int outputLengthWordLookup = 50;
			int classNum = 2;
			int embeddingLengthItemLookup = 75;

			int roundNum = 1500;
			double probThreshold = 0.001;
			double learningRate = 0.03;
			double randomizeBase = 0.01;


			List<View> views=new ArrayList<View>();
			views.add(new View(view1Name,embeddingFileView1,embeddingLength,trainIds,testIds,allDataView1));
			views.add(new View(view2Name,embeddingFileView2,embeddingLength,trainIds,testIds,allDataView2));
			//views.add(new View(view3Name,embeddingFileView3,embeddingLength,trainIds,testIds,allDataView3));

			List<Integer> windowSizeWordLookupList=new ArrayList<Integer>();
			//windowSizeWordLookupList.add(1);
			windowSizeWordLookupList.add(2);
			windowSizeWordLookupList.add(3);

			TestTool.println("Start...");
			//setLogPath
			destName="";
			if(views.size()==1){
				destName=views.get(0).viewName;
			}
			if(views.size()>1){
				for(int i=0;i<views.size();++i){
					if(i<views.size()-1)
						destName+=views.get(i).viewName+"+";
					else
						destName+=views.get(i).viewName;
				}
			}
			GlobleLog.setLogDir(FileTool.backReplaceDirNode(destBase+"5fold\\"+fold+"\\", "5fold", destName));
			Mynn7 main = new Mynn7(
					views,
					windowSizeWordLookupList,
					outputLengthWordLookup,
					embeddingLengthItemLookup,
					classNum, 
					randomizeBase);
			Double maxAcc=main.run(roundNum, 
					probThreshold, 
					learningRate, 
					classNum,
					"");
			accs.add(maxAcc);
			TestTool.println("End");
		}
		double avgAcc=0;
		for(Double acc:accs){
			avgAcc+=acc;
		}
		avgAcc/=accs.size();
		GlobleLog.setLogDir(destBase+destName+"\\");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+accs.get(f));
		}
		TestTool.println(totalFold+" fold avg acc="+avgAcc+"\r\n");
	}

}
