package wyConnect;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import adapter.WYIO;
import model.DocAverage;
import nnet.AverageLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.MultiConnectLayer;
import nnet.NNInterface;
import other.Data;
import other.Funcs;

public abstract class Net4ViewUser {
	public HashMap<String, Integer> wordVocab = null;
	public String unkStr = "</s>";
	
	public final static String TRAIN="train";
	public final static String TEST="test";
	public final static String VALIDATION="validation";
	
	public List<Data> trainDataList=null;
	public List<Data> validationDataList=null;
	public List<Data> testDataList=null;  
	
	public MultiConnectLayer connect;
	public AverageLayer average;
	
	public LookupLayer viewLookup;
	
	/////
	public int idxData;
	public String type;
	
    public void randomize(Random r, double min, double max){
		
	}
    
	public abstract List<NNInterface> getDocAverageList(int[][]  wordIdMatrix) throws Exception;
	
	public List<NNInterface> getDocAverageList(int idxData,String type) throws Exception{
		this.idxData=idxData;
		this.type=type;
		return getDocAverageList(this.getWordIdMatrix(idxData,type));
	}
	
	public abstract int[][] getWordIdMatrix(int idxData,String type);
	
	public Data getData(){
		Data data=null;
		if(type.equals(Net4ViewUser.TRAIN)){
			data= this.trainDataList.get(this.idxData);
		}
		if(type.equals(Net4ViewUser.TEST)){
			data= this.testDataList.get(this.idxData);
		}
		if(type.equals(Net4ViewUser.VALIDATION)){
			data=this.validationDataList.get(this.idxData);
		}
		return data;
	}
	
	public void loadData(
			String trainFile,
			String testFile)
	{
		System.out.println("================ start loading corpus ==============");
		trainDataList = new ArrayList<Data>();  
		//		
		Funcs.loadCorpus(trainFile, "utf8", trainDataList);
		testDataList = new ArrayList<Data>();  
		Funcs.loadCorpus(testFile, "utf8", testDataList);

		System.out.println("training size: " + trainDataList.size());
		System.out.println("testDataList size: " + testDataList.size());
		System.out.println("================ finsh loading corpus ==============");
	}
	public void loadData(String trainIds,String validationIds,String testIds,String allReviews) throws IOException{
		System.out.println("================ start loading corpus ==============");
		trainDataList = new ArrayList<Data>();  
		WYIO.readCorpus(trainIds, "\t", allReviews,"\t\t",trainDataList);

		testDataList = new ArrayList<Data>();  
		WYIO.readCorpus(testIds, "\t",allReviews,"\t\t",testDataList);
		
		validationDataList=new ArrayList<Data>();
		WYIO.readCorpus(validationIds, "\t",allReviews,"\t\t",validationDataList);

		System.out.println("training size: " + trainDataList.size());
		System.out.println("validation size: " + validationDataList.size());
		System.out.println("test size: " + testDataList.size());
		System.out.println("================ finsh loading corpus ==============");
		
	}
}
