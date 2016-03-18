package wyConnect;

import java.io.IOException;
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
import adapter.WYIO;
import model.DocAverage;
import nnet.AverageLayer;
import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.MultiConnectLayer;
import nnet.NNInterface;
import nnet.SoftmaxLayer;
import other.Data;
import other.Funcs;
import other.Metric;

public class CNNME4ViewUser extends Net4ViewUser{
	public List<LookupLinearTanh> xseedLLTxList;

	public LookupAverage lookupAverageForME;

	////////////////
	public View view;

	public CNNME4ViewUser(View view,List<Integer> windowSizeLookupList,int outputLengthWordLookup) throws Exception{
		this(view,view.embeddingFileWord,view.embeddingLengthWord,windowSizeLookupList,outputLengthWordLookup,view.trainIds,view.validationIds,view.testIds,view.allDatas);
		
	}
	public CNNME4ViewUser(
			String embeddingFileWord,
			int embeddingLengthWord,
			List<Integer> windowSizeLookupList,
			List<Integer> outputLengthWordLookupList,
			int averageOutputLength,
			String trainIds,
			String validationIds,
			String testIds,
			String allReviews
			) throws Exception{

		init(embeddingFileWord,
				embeddingLengthWord,
				windowSizeLookupList,
				outputLengthWordLookupList,
				averageOutputLength,
				trainIds,
				validationIds,
				testIds,
				allReviews
				);
	}
	public CNNME4ViewUser(
			View view,
			String embeddingFileWord,
			int embeddingLengthWord,
			List<Integer> windowSizeLookupList,
			int outputLengthWordLookup,
			String trainIds,
			String validationIds,
			String testIds,
			String allReviews
			) throws Exception{
		this.view=view;
		List<Integer> outputLengthWordLookupList=new ArrayList<Integer>();
		for(int i=0;i<windowSizeLookupList.size();++i){
			outputLengthWordLookupList.add(outputLengthWordLookup);
		}
		init(embeddingFileWord,
				embeddingLengthWord,
				windowSizeLookupList,
				outputLengthWordLookupList,
				outputLengthWordLookup,
				trainIds,
				validationIds,
				testIds,
				allReviews
				);

	}
	public void init(
			String embeddingFileWord,
			int embeddingLengthWord,
			List<Integer> windowSizeLookupList,
			List<Integer> outputLengthWordLookupList,
			int averageOutputLength,
			String trainIds,
			String validationIds,
			String testIds,
			String allReviews
			) throws Exception{
		wordVocab = new HashMap<String, Integer>();

		int embeddingLineCount = Funcs.lineCounter(embeddingFileWord, "utf8")-1;
		double[][] table = new double[embeddingLineCount][];
		Funcs.loadEmbeddingFile(embeddingFileWord, embeddingLengthWord, "utf8", 
				false, wordVocab, table);

		if(windowSizeLookupList.size()!=outputLengthWordLookupList.size()){
			throw new Exception("windowSizeLookupList.size()!=outputLengthWordLookupList.size()");
		}

		boolean isAll=false;
		if(view.netType==2)
		{
			isAll=true;
		}

		if(isAll||view.netType==1){
			//CNN
			xseedLLTxList=new ArrayList<LookupLinearTanh>();
			for(int i=0;i<windowSizeLookupList.size();++i){
				LookupLinearTanh xseedLLTx=new LookupLinearTanh(windowSizeLookupList.get(i), wordVocab.size(), outputLengthWordLookupList.get(i), embeddingLengthWord);
				xseedLLTx.lookup.setEmbeddings(table);
				xseedLLTxList.add(xseedLLTx);
			}
			int[] mutConInputLengths=new int[outputLengthWordLookupList.size()+1];
			int i=0;
			for(;i<outputLengthWordLookupList.size();++i){
				mutConInputLengths[i]=outputLengthWordLookupList.get(i);
			}
			viewLookup= new LookupLayer(100,1, 1);
			mutConInputLengths[i]=viewLookup.output.length;

			connect = new MultiConnectLayer(mutConInputLengths);

			average = new AverageLayer(connect.outputLength, averageOutputLength);
			connect.link(average);
		}

		if(isAll||view.netType==0){
			//ME(LR)
			this.lookupAverageForME=new LookupAverage(embeddingLengthWord,wordVocab.size());
			this.lookupAverageForME.setEmbeddings(table);
		}


		this.loadData(trainIds,validationIds,testIds, allReviews);
	}
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(LookupLinearTanh llt:xseedLLTxList){
			llt.randomize(r, min, max);
		}
		viewLookup.randomize(r, min, max);
	}

	public List<NNInterface> getDocAverageList(int[][]  wordIdMatrix) throws Exception{
		int unkIdx=wordVocab.get(unkStr);
		List<NNInterface> docAverageList=new ArrayList<NNInterface>();
		for(LookupLinearTanh llt:xseedLLTxList){
			docAverageList.add(new DocAverage(llt,wordIdMatrix,unkIdx));
		}
		return docAverageList;
	}

	public int[][] getWordIdMatrix(int idxData,String type){
		//String sentenceSplit="。{1,}|！{1,}|？{1,}|\\n";
		String sentenceSplit="\\n";
		Data data = null;
		if(type.equals("train")){
			data= this.trainDataList.get(idxData);
		}
		if(type.equals("test")){
			data= this.testDataList.get(idxData);
		}
		if(type.equals("validation")){
			data=this.validationDataList.get(idxData);
		}

		String[] sentences = data.reviewText.split(sentenceSplit);
		int[][] wordIdMatrix = Funcs.fillDocument(sentences, this.wordVocab, this.unkStr);
		return wordIdMatrix;
	}
	public void update(double learningRate) {
		// TODO Auto-generated method stub

	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub

	}

	public void clearGrad() {
		// TODO Auto-generated method stub

	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub

	}

	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub

	}

	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		return null;
	}

}
