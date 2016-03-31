package wyModel;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import adapter.FileTool;
import adapter.GlobleLog;
import adapter.TestTool;
import adapter.ToolKit;
import adapter.WYIO;
import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.MultiConnectLayer;
import nnet.NNInterface;
import nnet.SoftmaxLayer;
import other.Data;
import other.Metric;
import wyConnect.CNNME4ViewUser;
import wyConnect.LinearTanh;
import wyConnect.MetricResult;
import wyConnect.Net4ViewUser;
import wyConnect.PredictData;
import wyConnect.PredictResult;
import wyConnect.ResultBean;
import wyConnect.View;
import wyConnect.ViewInfo;

public class Mynnme_1linear_auto {
	List<CNNME4ViewUser> net4UserList;

	LookupLayer fusionLookup;
	MultiConnectLayer connectFusion;

	LinearLayer linearForSoftmax;

	SoftmaxLayer softmax;

	public Mynnme_1linear_auto(
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
		net4UserList=new ArrayList<CNNME4ViewUser>();
		for(View view:views){
			CNNME4ViewUser net4User=new CNNME4ViewUser(view,windowSizeWordLookupList,outputLengthWordLookup);
			net4UserList.add(net4User);
		}  

		//fusion lookup layers
		fusionLookup = new LookupLayer(embeddingLengthItemLookup,1, 1);
		int mclength=1;
		for(CNNME4ViewUser nn:net4UserList)
		{
			if(nn.view.netType==2)
			{
				mclength+=2;
			}
			else
			{
				mclength+=1;
			}
		}
		int[] multConInputlengths=new int[mclength];
		int k=0;
		for(int i=0;i<net4UserList.size();++i){
			switch(net4UserList.get(i).view.netType)
			{
			case 0:
				multConInputlengths[k++]=net4UserList.get(i).lookupAverageForME.average.outputLength;
				break;
			case 1:
				multConInputlengths[k++]=net4UserList.get(i).average.outputLength;
				break;
			case 2:
				multConInputlengths[k++]=net4UserList.get(i).average.outputLength;
				multConInputlengths[k++]=net4UserList.get(i).lookupAverageForME.average.outputLength;
				break;
			}
		}

		multConInputlengths[k]=fusionLookup.output.length;

		connectFusion = new MultiConnectLayer(multConInputlengths);
		k=0;
		for(int i=0;i<net4UserList.size();++i){
			switch(net4UserList.get(i).view.netType)
			{
			case 0:
				net4UserList.get(i).lookupAverageForME.average.link(connectFusion,k++);
				break;
			case 1:
				net4UserList.get(i).average.link(connectFusion, k++);
				break;
			case 2:
				net4UserList.get(i).average.link(connectFusion, k++);
				net4UserList.get(i).lookupAverageForME.average.link(connectFusion,k++);
				break;
			}
		}
		fusionLookup.link(connectFusion, k);

		// linear for softmax
		linearForSoftmax = new LinearLayer(connectFusion.outputLength, classNum);
		connectFusion.link(linearForSoftmax);


		softmax = new SoftmaxLayer(linearForSoftmax.outputLength);
		linearForSoftmax.link(softmax);

		Random rnd = new Random(); 
		for(CNNME4ViewUser net4User:net4UserList){
			int netType=net4User.view.netType;
			if(netType==1||netType==2)
			{
				net4User.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
			}
		}
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
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

		int maxTestAccRound=-1;
		PredictResult maxTestAcc=new PredictResult();

		PredictResult prstTestByMeanErrorVal=null;
		PredictResult prstTestByAccVal=null;

		int minMeanErrorValRound=-1;
		PredictResult minMeanErrorVal=new PredictResult();
		int maxAccValRound=-1;
		PredictResult maxAccVal=new PredictResult();


		for(int round = 1; round <= roundNum; round++)
		{
			System.out.println("============== running round: " + round + " ===============");

			for(CNNME4ViewUser n4u:net4UserList){
				Collections.shuffle(n4u.trainDataList, new Random(1000+round*2));
			}

			for(int idxData = 0; idxData < net4UserList.get(0).trainDataList.size(); idxData++)
			{
				Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap=new LinkedHashMap<CNNME4ViewUser,List<NNInterface>>();
				for(CNNME4ViewUser net4User:net4UserList){
					boolean isAll=false;
					if(net4User.view.netType==2)
					{
						isAll=true;
					}
					if(isAll||net4User.view.netType==0)
					{
						net4User.lookupAverageForME.setInputIdAAndLinkLA(net4User.getWordIdMatrix(idxData, "train"));
					}
					List<NNInterface> docAverageList=null;
					if(isAll||net4User.view.netType==1)
					{
						docAverageList=net4User.getDocAverageList(idxData,"train");
					}
					viewUserDocAverageListMap.put(net4User, docAverageList);
				}

				if(!this.idDocAverageListValid(viewUserDocAverageListMap))
				{
					continue;
				}


				fusionLookup.input[0] =0;

				// important
				this.net4ViewUserListLinkAndForward(viewUserDocAverageListMap);
				fusionLookup.forward();

				connectFusion.forward();

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

				connectFusion.backward();
				fusionLookup.backward();

				this.net4ViewUserListBackward(viewUserDocAverageListMap);

				// update
				linearForSoftmax.update(learningRate);
				this.net4ViewUserListUpdate(viewUserDocAverageListMap, learningRate);
				fusionLookup.update(learningRate);


				// clearGrad
				this.net4ViewUserListClearGrad(viewUserDocAverageListMap);

				connectFusion.clearGrad();
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

			System.out.println("============= finish training round: " + round + " ==============");

			PredictResult testResult=predict(round,Net4ViewUser.TEST);

			if(testResult.acc>maxTestAcc.acc)
			{
				maxTestAcc=testResult;
				maxTestAccRound=round;
			}

			PredictResult validationResult=predict(round,Net4ViewUser.VALIDATION);
			if(validationResult.acc>maxAccVal.acc)
			{
				maxAccVal=validationResult;
				prstTestByAccVal=testResult;
				maxAccValRound=round;
			}
			if(validationResult.meanError<minMeanErrorVal.meanError)
			{
				minMeanErrorVal=validationResult;
				prstTestByMeanErrorVal=testResult;
				minMeanErrorValRound=round;
			}
		}
		TestTool.println("=============total "+roundNum+" epochs"+" =====================");
		TestTool.println("MaxValidationAcc="+maxAccVal.acc);
		TestTool.println("MaxValidationAccRound="+maxAccValRound);
		TestTool.println("TestAcc="+prstTestByAccVal.acc);
		TestTool.println("TestMacroF="+prstTestByAccVal.macroF);

		TestTool.println("==========================");
		TestTool.println("MinValidationError="+minMeanErrorVal.meanError);
		TestTool.println("MinValidationErrorRound="+minMeanErrorValRound);
		TestTool.println("TestAcc="+prstTestByMeanErrorVal.acc);
		TestTool.println("TestMacroF="+prstTestByMeanErrorVal.macroF);

		TestTool.println("==========================");
		TestTool.println("MaxTestAcc="+maxTestAcc.acc);
		TestTool.println("TestMacroF="+maxTestAcc.macroF);
		TestTool.println("MaxTestAccRound="+maxTestAccRound);

		TestTool.println("========================================================== finnish in "+new Date()+"============================================================");
		ResultBean roundResult=new ResultBean();
		roundResult.setPreTestByAccVal(prstTestByAccVal, maxAccValRound);
		roundResult.setPreTestByMeanErrorVal(prstTestByMeanErrorVal, minMeanErrorValRound);
		roundResult.setMaxAccTest(maxTestAcc, maxTestAccRound);
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
		PredictResult result=new PredictResult();
		for(int idxData = 0; idxData < oneViewDataList.size(); idxData++)
		{
			Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap=new LinkedHashMap<CNNME4ViewUser,List<NNInterface>>();
			for(CNNME4ViewUser net4User:net4UserList){
				boolean isAll=false;
				if(net4User.view.netType==2)
				{
					isAll=true;
				}
				if(isAll||net4User.view.netType==0)
				{
					net4User.lookupAverageForME.setInputIdAAndLinkLA(net4User.getWordIdMatrix(idxData, type));
				}
				List<NNInterface> docAverageList=null;
				if(isAll||net4User.view.netType==1)
				{
					docAverageList=net4User.getDocAverageList(idxData,type);
				}
				viewUserDocAverageListMap.put(net4User, docAverageList);
			}

			if(!this.idDocAverageListValid(viewUserDocAverageListMap))
			{
				continue;
			}


			fusionLookup.input[0] =0;

			// important
			this.net4ViewUserListLinkAndForward(viewUserDocAverageListMap);

			fusionLookup.forward();
			connectFusion.forward();

			linearForSoftmax.forward();
			softmax.forward();

			PredictData predictdata=new PredictData();
			List<Double> probs=new ArrayList<Double>();
			int predClass = -1;
			double maxPredProb = -1.0;
			for(int ii = 0; ii < softmax.length; ii++)
			{
				probs.add(softmax.output[ii]);
				if(softmax.output[ii] > maxPredProb)
				{
					maxPredProb = softmax.output[ii];
					predClass = ii;
				}
			}
			predList.add(predClass + 1);
			Data data=oneViewDataList.get(idxData);
			goldList.add(data.goldRating);

			data.predictedRating=predClass + 1;
			predictdata.setCateProbls(probs);
			predictdata.data=data;

			result.add(predictdata);
			// set cross-entropy error 
			// we minus 1 because the saved goldRating is in range 1~5, while what we need is in range 0~4
			int goldRating = data.goldRating - 1;
			lossV += -Math.log(softmax.output[goldRating]);
			lossC += 1;
		}
		MetricResult metricResult=Metric.calcMetric(goldList, predList);
		double meanError=lossV/lossC*1.0 ;
		result.acc=metricResult.acc;
		result.macroF=metricResult.macroF;
		result.meanError=meanError;

		TestTool.println("lossV/lossC = " + lossV + "/" + lossC + "\t"+ " = " + meanError);
		TestTool.println(type+":============== finish predicting =================\r\n");
		return result;
	}

	public void net4ViewUserListLinkAndForward(Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap) throws Exception{
		for(Map.Entry<CNNME4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNNME4ViewUser net4User=entry.getKey();
			boolean isAll=false;
			if(net4User.view.netType==2)
			{
				isAll=true;
			}
			if(isAll||net4User.view.netType==1)
			{
				List<NNInterface> docAverageList=entry.getValue();
				//link
				int i=0;
				for(NNInterface docAverage: docAverageList){
					docAverage.link(net4User.connect, i);
					++i;
				}

				net4User.viewLookup.link(net4User.connect, i);

				//cnn forward
				for(NNInterface docAverage: docAverageList){
					docAverage.forward();
				}
				net4User.viewLookup.forward();
				net4User.connect.forward();
				net4User.average.forward();
			}

			if(isAll||net4User.view.netType==0)
			{
				//ME forward
				net4User.lookupAverageForME.forward();
			}
		}
	}

	public void net4ViewUserListBackward(Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNNME4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNNME4ViewUser net4User=entry.getKey();
			int netType=net4User.view.netType;
			if(netType==1||netType==2)
			{
				List<NNInterface> docAverageList=entry.getValue();

				net4User.average.backward();
				net4User.connect.backward();
				net4User.viewLookup.backward();
				for(NNInterface docAverage: docAverageList){
					docAverage.backward();
				}
			}
		}
	}

	public void net4ViewUserListUpdate(Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap,double learningRate){
		for(Map.Entry<CNNME4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNNME4ViewUser net4User=entry.getKey();
			int netType=net4User.view.netType;

			if(netType==1||netType==2)
			{
				List<NNInterface> docAverageList=entry.getValue();

				net4User.viewLookup.update(learningRate);
				for(NNInterface docAverage:docAverageList){
					docAverage.update(learningRate);
				}
			}
		}
	}

	public void net4ViewUserListClearGrad(Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNNME4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNNME4ViewUser net4User=entry.getKey();

			int netType=net4User.view.netType;
			if(netType==1||netType==2)
			{
				List<NNInterface> docAverageList=entry.getValue();

				for(NNInterface docAverage:docAverageList){
					docAverage.clearGrad();
				}
				net4User.viewLookup.clearGrad();

				net4User.connect.clearGrad();
				net4User.average.clearGrad();
			}
		}
	}
	protected boolean idDocAverageListValid(Map<CNNME4ViewUser,List<NNInterface>> viewUserDocAverageListMap){
		for(Map.Entry<CNNME4ViewUser,List<NNInterface>> entry:viewUserDocAverageListMap.entrySet()){
			CNNME4ViewUser net4User=entry.getKey();
			List<NNInterface> docAverageList=entry.getValue();

			if(docAverageList!=null)
			{
				for(NNInterface docAverage:docAverageList){
					if(!docAverage.isValid()){
						System.out.println(net4User.getData().toString()+" docAverage.sentenceConvList.size() == 0");
						return false;
					}
				}
			}
		}
		return true;
	}
	public static void expRun(String sourceBase,String destBase,List<ViewInfo> viewInfos,int roundNum,int totalFold) throws Exception{

		String destName="";
		List<ResultBean> results=new ArrayList<ResultBean>();
		//int totalFold=5;
		for(int fold=0;fold<totalFold;++fold){
			String trainIds = sourceBase+"5fold\\"+fold+"\\train.txt";
			String validationIds=sourceBase+"5fold\\"+fold+"\\validation.txt";
			String testIds =sourceBase+"5fold\\"+fold+"\\test.txt";

			int outputLengthWordLookup = 85;

			int classNum = 2;
			int embeddingLengthItemLookup = 100;

			//int roundNum = 100;
			double probThreshold = 0.001;
			double learningRate = 0.03;
			double randomizeBase = 0.01;


			List<View> views=new ArrayList<View>();
			for(ViewInfo vinfo:viewInfos){
				views.add(new View(vinfo,vinfo.netType,trainIds,validationIds,testIds));
			}

			List<Integer> windowSizeWordLookupList=new ArrayList<Integer>();
			windowSizeWordLookupList.add(1);
			windowSizeWordLookupList.add(2);
			windowSizeWordLookupList.add(3);

			TestTool.println("Start...");
			//setLogPath
			destName="";
			String netType="";
			String concateStr="_";
			if(views.size()==1){
				destName=views.get(0).viewName+concateStr+getNetType(views.get(0).netType);
			}
			if(views.size()>1){
				for(int i=0;i<views.size();++i){
					if(i<views.size()-1)
						destName+=views.get(i).viewName+concateStr+getNetType(views.get(i).netType)+"+";
					else
						destName+=views.get(i).viewName+concateStr+getNetType(views.get(i).netType);
				}
			}
			String logDir=FileTool.backReplaceDirNode(destBase+"5fold\\"+fold+"\\", "5fold", destName);
			GlobleLog.setLogDir(logDir);

			Mynnme_1linear_auto main = new Mynnme_1linear_auto(
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
			
			//write tesing_id and result_cnn for each fold exp
			WYIO.writePredictResult(logDir,result.preTest_accVal,"accVal");
			WYIO.writePredictResult(logDir,result.preTest_meanErrorVal,"mErrorVal");
		}

		double avgAcc_maxAccValidation=0;
		double avgMacroF_maxAccVal=0;

		double avgAcc_minErrorValidation=0;
		double avgMacroF_minErrorVal=0;

		double avgAccmax=0;
		double avgMacroF_maxTestAcc=0;
		for(ResultBean result:results){
			avgAcc_maxAccValidation+=result.preTest_accVal.acc;
			avgMacroF_maxAccVal+=result.preTest_accVal.macroF;

			avgAcc_minErrorValidation+=result.preTest_meanErrorVal.acc;
			avgMacroF_minErrorVal+=result.preTest_meanErrorVal.macroF;

			avgAccmax+=result.maxAccTest.acc;
			avgMacroF_maxTestAcc+=result.maxAccTest.macroF;

		}
		avgAcc_maxAccValidation/=results.size();
		avgAcc_minErrorValidation/=results.size();
		avgAccmax/=results.size();

		GlobleLog.setLogDir(destBase+destName+"\\");

		TestTool.println("======Acc Max===========");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+results.get(f).maxAccTest.acc+" macroF="+results.get(f).maxAccTest.macroF
					+ " round="+results.get(f).maxAccRound);
		}
		TestTool.println(totalFold+" fold avg acc="+avgAccmax+"\r\n");


		TestTool.println("======Acc stop by maxAccValidation===========");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+results.get(f).preTest_accVal.acc+" macroF="+results.get(f).preTest_accVal.macroF
					+" round="+results.get(f).MaxAccValidationRound);
		}
		TestTool.println(totalFold+" fold avg acc="+avgAcc_maxAccValidation+"\r\n");


		TestTool.println("======Acc stop by minErrorValidation===========");
		for(int f=0;f<totalFold;++f){
			TestTool.println("fold "+f+" "+destName+" acc="+results.get(f).preTest_meanErrorVal.acc+" macroF="+results.get(f).preTest_meanErrorVal.macroF
					+" round="+results.get(f).minErrorValidationRound);
		}
		TestTool.println(totalFold+" fold avg acc="+avgAcc_minErrorValidation+"\r\n");
	}
	public static String getNetType(int nt)
	{
		String netType="";
		switch(nt)
		{
		case 0:
			netType="noCNN";
			break;
		case 1:
			netType="CNN";
			break;
		case 2:
			netType="CNNnoCNN";
			break;
		}
		return netType;
	}

	
	
	public static void main(String[] args) throws Exception{
		String sourceBase="F:\\ExpData\\DataIntegate\\source\\nne\\publicinfo4\\zps-400-expset\\FTnum3_-1\\";
		String destBase=FileTool.forwardReplaceDirNode(sourceBase, "source", "dest");
		destBase=destBase+"Test/";

		List<List<ViewInfo>> expViewInfos=new ArrayList<List<ViewInfo>>();


		//		List<ViewInfo> viewD=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"tag",viewD);
		//		expViewInfos.add(viewD);

		//		List<ViewInfo> viewE=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"id_w2v",viewE);
		//		expViewInfos.add(viewE);
		//
		//		List<ViewInfo> viewsF=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"id_avgT",viewsF);
		//		expViewInfos.add(viewsF);

		//		List<ViewInfo> viewsG=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"weibo(10x20)",0,viewsG);
		//		addExpView(sourceBase,"tag",0,viewsG);
		//		addExpView(sourceBase,"id_w2v_sort_cluster(step10)",0,viewsG);
		//		expViewInfos.add(viewsG);
		//
		//		List<ViewInfo> viewsH=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"weibo(10x20)",1,viewsH);
		//		addExpView(sourceBase,"tag",0,viewsH);
		//		addExpView(sourceBase,"id_w2v_sort_cluster(step10)",0,viewsH);
		//		expViewInfos.add(viewsH);
		//
		
//				List<ViewInfo> viewsI=new ArrayList<ViewInfo>();
//				addExpView(sourceBase,"tag_sort(oldVec)",0,viewsI);
//				expViewInfos.add(viewsI);

		//		List<ViewInfo> viewsJ=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"tag(negative)",0,viewsJ);
		//		expViewInfos.add(viewsJ);

		//		List<ViewInfo> viewsK=new ArrayList<ViewInfo>();
		//		addExpView(sourceBase,"weibo(10x20)(Wm.xId_w2v)",viewsK);
		//		expViewInfos.add(viewsK);

		int roundNum=100;
		int totalFold=2;
		for(List<ViewInfo> viewInfos:expViewInfos){
			expRun(sourceBase,destBase,viewInfos,roundNum,totalFold);
		}
	}

	public static void addExpView(String sourceBase,String viewName,int netType,List<ViewInfo> views) throws IOException
	{
		String embeddingFileView =sourceBase+viewName+"\\word_vector.txt";
		String allDataView=sourceBase+viewName+"\\allReview.txt";

		int embeddingLengthView =WYIO.readVecDiem(embeddingFileView, " ");

		views.add(new ViewInfo(viewName,embeddingFileView,embeddingLengthView,allDataView,netType));
	}

}
