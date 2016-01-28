package wyModel;

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
import wyConnect.CNN4ViewUser;
import wyConnect.LinearTanh;
import wyConnect.Net4ViewUser;
import wyConnect.PredictResult;
import wyConnect.ResultBean;
import wyConnect.View;

public class Mynn_2linear {
	List<CNN4ViewUser> net4UserList;

	LookupLayer fusionLookup;
	MultiConnectLayer connectFusion;

	LinearLayer linear1;
	LinearLayer linear2;

	SoftmaxLayer softmax;

	public Mynn_2linear(
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
		int hiddenNum=connectFusion.outputLength/4;
		linear1 = new LinearLayer(connectFusion.outputLength, hiddenNum);
		connectFusion.link(linear1);

		linear2 = new LinearLayer(linear1.outputLength, classNum);
		softmax = new SoftmaxLayer(linear2.outputLength);
		linear2.link(softmax);

		Random rnd = new Random(); 
		for(CNN4ViewUser net4User:net4UserList){
			net4User.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		}
		linear1.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		linear2.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
	}

	public ResultBean run(
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

		List<Double> accsTest=new ArrayList<Double>();
		List<Double> accsValidation=new ArrayList<Double>();
		List<Double> meanErrorsValidation=new ArrayList<Double>();
		for(CNN4ViewUser n4u:net4UserList){
			Collections.shuffle(n4u.trainDataList, new Random(1000));
		}
		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");

			for(int idxData = 0; idxData < net4UserList.get(0).trainDataList.size(); idxData++)
			{
				Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap=new LinkedHashMap<CNN4ViewUser,List<NNInterface>>();
				for(CNN4ViewUser net4User:net4UserList){
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

				linear1.forward();
				linear2.forward();
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
				linear2.backward();
				linear1.backward();

				connectFusion.backward();
				fusionLookup.backward();

				this.net4ViewUserListBackward(viewUserDocAverageListMap);

				// update
				linear2.update(learningRate);
				linear1.update(learningRate);
				this.net4ViewUserListUpdate(viewUserDocAverageListMap, learningRate);
				fusionLookup.update(learningRate);


				// clearGrad
				this.net4ViewUserListClearGrad(viewUserDocAverageListMap);

				connectFusion.clearGrad();
				fusionLookup.clearGrad();

				linear1.clearGrad();
				linear2.clearGrad();
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
            
			PredictResult testResult=predict(round,Net4ViewUser.TEST);
			double accTest=testResult.acc;
			accsTest.add(accTest);
			
			PredictResult validationResult=predict(round,Net4ViewUser.VALIDATION);
			accsValidation.add(validationResult.acc);
			meanErrorsValidation.add(validationResult.meanError);
		}
		
		Map.Entry<Integer,Double> maxValueMap=ToolKit.getMaxValue(accsValidation);
		TestTool.println("=============total "+roundNum+" epochs"+" =====================");
		TestTool.println("MaxValidationAcc="+maxValueMap.getValue());
		TestTool.println("MaxValidationAccRound="+(maxValueMap.getKey()+1));
		TestTool.println("TestAcc="+accsTest.get(maxValueMap.getKey()));
		
		TestTool.println("==========================");
		Map.Entry<Integer,Double> minValueMap=ToolKit.getMinValue(meanErrorsValidation);
		TestTool.println("MinValidationError="+minValueMap.getValue());
		TestTool.println("MinValidationErrorRound="+(minValueMap.getKey()+1));
		TestTool.println("TestAcc="+accsTest.get(minValueMap.getKey()));
		
		TestTool.println("==========================");
		Map.Entry<Integer,Double> maxTestValueMap=ToolKit.getMaxValue(accsTest);
		TestTool.println("MaxTestAcc="+maxTestValueMap.getValue());
		TestTool.println("MaxTestAccRound="+(maxTestValueMap.getKey()+1));
		
		//求验证集上的极小点
//		List<Double> delatMeanErrorVallidation=new ArrayList<Double>();
//		for(int i=1;i<meanErrorsValidation.size();++i){
//			delatMeanErrorVallidation.add(meanErrorsValidation.get(i)-meanErrorsValidation.get(i-1));
//		}
//		TestTool.println("==========================");
//		Map.Entry<Integer,Double> deltaMeanErrorMap=ToolKit.getMinimalPoint(delatMeanErrorVallidation, 0.01);
//		TestTool.println("MaxTestAcc="+maxTestValueMap.getValue());
//		TestTool.println("MaxTestAccRound="+(maxTestValueMap.getKey()+1));
		TestTool.println("========================================================== finnish in "+new Date()+"============================================================");
		ResultBean roundResult=new ResultBean();
		
		roundResult.acc_MaxAccValidation=accsTest.get(maxValueMap.getKey());
		roundResult.MaxAccValidationRound=maxValueMap.getKey()+1;
		roundResult.acc_minErrorValidation=accsTest.get(minValueMap.getKey());
		roundResult.minErrorValidationRound=minValueMap.getKey()+1;
		roundResult.acc_max=maxTestValueMap.getValue();
		roundResult.maxAccRound=maxTestValueMap.getKey()+1;
		return roundResult;
	}

	public PredictResult predict(int round,String type) throws Exception
	{
		TestTool.println(type+":=========== predicting round: " + round + " ===============");

		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();

		List<Data> oneViewDataList=null;
		if(type.equals(Net4ViewUser.TEST)){
			oneViewDataList=net4UserList.get(0).testDataList;
		}
		else if(type.equals(Net4ViewUser.VALIDATION)) {
			oneViewDataList=net4UserList.get(0).validationDataList;
		}
		else{
			oneViewDataList=net4UserList.get(0).trainDataList;
		}
		
        double lossV=0;
        int lossC=0;
		for(int idxData = 0; idxData < oneViewDataList.size(); idxData++)
		{
			Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap=new LinkedHashMap<CNN4ViewUser,List<NNInterface>>();
			for(CNN4ViewUser net4User:net4UserList){
				List<NNInterface> docAverageList=net4User.getDocAverageList(idxData,type);
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

			linear1.forward();
			linear2.forward();
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
			Data data=oneViewDataList.get(idxData);
			goldList.add(data.goldRating);
			
			// set cross-entropy error 
			// we minus 1 because the saved goldRating is in range 1~5, while what we need is in range 0~4
			int goldRating = data.goldRating - 1;
			lossV += -Math.log(softmax.output[goldRating]);
			lossC += 1;
		}
		double acc=Metric.calcMetric(goldList, predList);
		double meanError=lossV/lossC*1.0 ;
		PredictResult result=new PredictResult(acc,meanError);
		TestTool.println("lossV/lossC = " + lossV + "/" + lossC + "\t"+ " = " + meanError);
		TestTool.println(type+":============== finish predicting =================\r\n");
		return result;
	}

	public void net4ViewUserListLinkAndForward(Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap) throws Exception{
		for(Map.Entry<CNN4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
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
			net4User.average.forward();

		}
	}

	public void net4ViewUserListBackward(Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			net4User.average.backward();
			net4User.connect.backward();
			net4User.viewLookup.backward();
			for(NNInterface docAverage: docAverageList){
				docAverage.backward();
			}
		}
	}

	public void net4ViewUserListUpdate(Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap,double learningRate){
		for(Map.Entry<CNN4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			net4User.viewLookup.update(learningRate);
			for(NNInterface docAverage:docAverageList){
				docAverage.update(learningRate);
			}
		}
	}

	public void net4ViewUserListClearGrad(Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			for(NNInterface docAverage:docAverageList){
				docAverage.clearGrad();
			}
			net4User.viewLookup.clearGrad();

			net4User.connect.clearGrad();
			net4User.average.clearGrad();
		}
	}
	protected boolean idDocAverageListValid(Map<CNN4ViewUser,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNN4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNN4ViewUser net4User=entry.getKey();
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
		

		String view1Name="Weibos-10x20";
		String embeddingFileView1 =sourceBase+view1Name+"\\wordVec\\word_vector.txt";
		String allDataView1=sourceBase+view1Name+"\\textData\\allReview.txt";
		int embeddingLengthView1 = 200;

		String view2Name="Tag-friTagVec";  
		String embeddingFileView2 =sourceBase+view2Name+"\\wordVec\\FriTag_vector.txt";
		String allDataView2=sourceBase+view2Name+"\\textData\\allReview.txt";
		int embeddingLengthView2 = 100;

		String view3Name="Tag-wordVec";  
		String embeddingFileView3 =sourceBase+view2Name+"\\wordVec\\word_vector.txt";
		String allDataView3=sourceBase+view2Name+"\\textData\\allReview.txt";
		int embeddingLengthView3 = 100;

		String destName="";

		List<ResultBean> results=new ArrayList<ResultBean>();
		int totalFold=5;
		for(int fold=0;fold<totalFold;++fold){
			String trainIds = sourceBase+"5fold\\"+fold+"\\train.txt";
			String validationIds=sourceBase+"5fold\\"+fold+"\\validation.txt";
			String testIds =sourceBase+"5fold\\"+fold+"\\test.txt";

			int outputLengthWordLookup = 50;
			int classNum = 2;
			int embeddingLengthItemLookup = 75;

			int roundNum = 500;
			double probThreshold = 0.001;
			double learningRate = 0.03;
			double randomizeBase = 0.01;


			List<View> views=new ArrayList<View>();
			views.add(new View(view1Name,embeddingFileView1,embeddingLengthView1,trainIds,validationIds,testIds,allDataView1));
			//views.add(new View(view2Name,embeddingFileView2,embeddingLengthView2,trainIds,validationIds,testIds,allDataView2));
			//views.add(new View(view3Name,embeddingFileView3,embeddingLengthView3,trainIds,validationIds,testIds,allDataView3));

			List<Integer> windowSizeWordLookupList=new ArrayList<Integer>();
			windowSizeWordLookupList.add(1);
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
			Mynn_2linear main = new Mynn_2linear(
					views,
					windowSizeWordLookupList,
					outputLengthWordLookup,
					embeddingLengthItemLookup,
					classNum, 
					randomizeBase);
			ResultBean result=main.run(roundNum, 
					probThreshold, 
					learningRate, 
					classNum,
					"");
			results.add(result);
			TestTool.println("End");
		}
		double avgAcc_maxAccValidation=0;
		double avgAcc_minErrorValidation=0;
		double avgAccmax=0;
		for(ResultBean result:results){
			avgAcc_maxAccValidation+=result.acc_MaxAccValidation;
			avgAcc_minErrorValidation+=result.acc_minErrorValidation;
			avgAccmax+=result.acc_max;
		}
		avgAcc_maxAccValidation/=results.size();
		avgAcc_minErrorValidation/=results.size();
		avgAccmax/=results.size();
		
		GlobleLog.setLogDir(destBase+destName+"\\");
		
		TestTool.println("======Acc Max===========");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+results.get(f).acc_max+" round="+results.get(f).maxAccRound);
		}
		TestTool.println(totalFold+" fold avg acc="+avgAccmax+"\r\n");
		
		
		TestTool.println("======Acc stop by maxAccValidation===========");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+results.get(f).acc_MaxAccValidation+" round="+results.get(f).MaxAccValidationRound);
		}
		TestTool.println(totalFold+" fold avg acc="+avgAcc_maxAccValidation+"\r\n");
		
		
		TestTool.println("======Acc stop by minErrorValidation===========");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+results.get(f).acc_minErrorValidation+" round="+results.get(f).minErrorValidationRound);
		}
		TestTool.println(totalFold+" fold avg acc="+avgAcc_minErrorValidation+"\r\n");
	}

}
