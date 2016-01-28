package wyConnect;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import model.DocAverage;
import nnet.AverageLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.MultiConnectLayer;
import nnet.NNInterface;
import other.Data;
import other.Funcs;

public class Avg4ViewUser extends Net4ViewUser{
	public LookupLayer lookuplayer;
	
	public int embeddingLengthWord;
	public double[][] table;
    
	public View view;
	
	public Avg4ViewUser(View view) throws IOException{
		this(view.embeddingFileWord,view.embeddingLengthWord,view.trainIds,view.testIds,view.allDatas);
		this.view=view;
	}
	public Avg4ViewUser(
			String embeddingFileWord,
			int embeddingLengthWord,
			String trainIds,
			String testIds,
			String allReviews
			) throws IOException{

		wordVocab = new HashMap<String, Integer>();

		int embeddingLineCount = Funcs.lineCounter(embeddingFileWord, "utf8")-1;
		table = new double[embeddingLineCount][];
		Funcs.loadEmbeddingFile(embeddingFileWord, embeddingLengthWord, "utf8", false, wordVocab, table);
		
		viewLookup= new LookupLayer(100,1, 1);
		average=new AverageLayer(embeddingLengthWord);

		this.loadData(trainIds, testIds, allReviews);
	}

	@Override
	public List<NNInterface> getDocAverageList(int[][] wordIdMatrix) throws Exception {
		// TODO Auto-generated method stub
		int unkIdx=wordVocab.get(unkStr);
		List<NNInterface> docAverageList=new ArrayList<NNInterface>();
		int k=0;
		
		lookuplayer=new LookupLayer(this.embeddingLengthWord,this.wordVocab.size(),wordIdMatrix[0].length);
		lookuplayer.setEmbeddings(this.table);
		
		for(int i=0;i<wordIdMatrix.length;++i){
			for(int j=0;j<wordIdMatrix[i].length;++j){
				lookuplayer.input[k]=wordIdMatrix[i][j];
				++k;
			}	
		}
		
		//link net
		connect=new MultiConnectLayer(new int[]{(lookuplayer.embeddingLength)*(lookuplayer.inputLength),viewLookup.output.length});
		
		average.initInput(connect.outputLength);
		
		docAverageList.add(lookuplayer);
		return docAverageList;
	}

	@Override
	public List<NNInterface> getDocAverageList(int idxData, String type) throws Exception {
		// TODO Auto-generated method stub
		this.idxData=idxData;
		this.type=type;
		return getDocAverageList(this.getWordIdMatrix(idxData,type));
	}

	@Override
	public int[][] getWordIdMatrix(int idxData, String type) {
		// TODO Auto-generated method stub
		String sentenceSplit="\\n";
		Data data = null;
		if(type.equals("train")){
			data= this.trainDataList.get(idxData);
		}
		if(type.equals("test")){
			data= this.testDataList.get(idxData);
		}
		
		String[] sentences = data.reviewText.split(sentenceSplit);
		int[][] wordIdMatrix = Funcs.fillDocument(sentences, this.wordVocab, this.unkStr);
		return wordIdMatrix;
	}

}
