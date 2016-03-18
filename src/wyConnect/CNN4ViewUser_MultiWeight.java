package wyConnect;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import model.DocAverage;
import nnet.AverageLayer;
import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.MultiConnectLayer;
import nnet.NNInterface;
import other.Data;
import other.Funcs;

public class CNN4ViewUser_MultiWeight extends Net4ViewUser{

	public List<LookupLinearTanh> xseedLLTxList;

	////////////////
	public View view;
	
	public LinearLayer viewMultiWeightLinerLayer;
    
	public CNN4ViewUser_MultiWeight(View view,List<Integer> windowSizeLookupList,int outputLengthWordLookup) throws Exception{
		this(view.embeddingFileWord,view.embeddingLengthWord,windowSizeLookupList,outputLengthWordLookup,view.trainIds,view.validationIds,view.testIds,view.allDatas);
		this.view=view;
	}
	public CNN4ViewUser_MultiWeight(
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
	public CNN4ViewUser_MultiWeight(
			String embeddingFileWord,
			int embeddingLengthWord,
			List<Integer> windowSizeLookupList,
			int outputLengthWordLookup,
			String trainIds,
			String validationIds,
			String testIds,
			String allReviews
			) throws Exception{
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
		
		viewMultiWeightLinerLayer=new LinearLayer(average.outputLength,average.outputLength);
		average.link(viewMultiWeightLinerLayer);
		
		this.loadData(trainIds,validationIds, testIds, allReviews);
	}
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		for(LookupLinearTanh llt:xseedLLTxList){
			llt.randomize(r, min, max);
		}
		viewLookup.randomize(r, min, max);
		viewMultiWeightLinerLayer.randomize(r, min, max);
	}

	public List<NNInterface> getDocAverageList(int[][]  wordIdMatrix) throws Exception{
		int unkIdx=wordVocab.get(unkStr);
		List<NNInterface> docAverageList=new ArrayList<NNInterface>();
		for(LookupLinearTanh llt:xseedLLTxList){
			docAverageList.add(new DocAverage(llt,wordIdMatrix,unkIdx));
		}
		return docAverageList;
	}
	public List<NNInterface> getDocAverageList(int idxData,String type) throws Exception{
		this.idxData=idxData;
		this.type=type;
		return getDocAverageList(this.getWordIdMatrix(idxData,type));
	}

	public int[][] getWordIdMatrix(int idxData,String type){
		String sentenceSplit="。{1,}|！{1,}|？{1,}|\\n";
		//String sentenceSplit="\\n";
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
