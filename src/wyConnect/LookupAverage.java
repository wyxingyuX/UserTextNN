package wyConnect;

import java.util.ArrayList;
import java.util.Random;

import javax.swing.event.ListSelectionEvent;

import nnet.AverageLayer;
import nnet.LookupLayer;
import nnet.NNInterface;

public class LookupAverage implements NNInterface{
	public LookupLayer lookup;
	public AverageLayer average;

	public LookupAverage() {
		// TODO Auto-generated constructor stub
	}
	public LookupAverage(int xEmbedingLength,int xVocabsize)
	{
		this.lookup=new LookupLayer();
		this.lookup.embeddingLength = xEmbedingLength;
		this.lookup.vocabSize = xVocabsize;
		
		this.lookup.table = new double[xVocabsize][];
		for (int i = 0; i < this.lookup.table.length; ++i)
        {
			this.lookup.table[i] = new double[this.lookup.embeddingLength];
        }
	
		this.lookup.adaLR = new double[xVocabsize][];
        for (int i = 0; i < this.lookup.adaLR.length; ++i)
        {
        	this.lookup.adaLR[i] = new double[xEmbedingLength];
        }
		
		int averageOutputLenth=xEmbedingLength;
		this.average=new AverageLayer(averageOutputLenth);
		// lookup and average be linked when inputLength is sured;
	}

	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}
	
	public void setEmbeddings(double[][] embed) throws Exception
    {
		this.lookup.setEmbeddings(embed);
    }
	
	public void setEmbeddingsShadow(double[][] embed)
	{
		this.lookup.table=embed;
	}
	
	public void setInputIdAAndLinkLA(int[] inputIds) throws Exception
	{
		this.lookup.inputLength=inputIds.length;
		this.lookup.input=inputIds;
		this.lookup.output= new double[this.lookup.embeddingLength * this.lookup.inputLength];
		this.lookup.outputG=new double[this.lookup.embeddingLength * this.lookup.inputLength];
		
		this.average.inputLength=this.lookup.embeddingLength * this.lookup.inputLength;
		this.average.input=new double[this.average.inputLength];
		this.average.inputG=new double[this.average.inputLength];
		
		
		this.lookup.link(this.average);
	}
	
	public void setInputIdAAndLinkLA(int[][] sentencesIds) throws Exception
	{
		ArrayList<Integer> inputIdsList=new ArrayList<Integer>();
		for(int i=0;i<sentencesIds.length;++i)
		{
			int[] sentence=sentencesIds[i];
			for(int j=0;j<sentence.length;++j)
			{
				inputIdsList.add(sentencesIds[i][j]);
			}
		}
		int[] inputIds=new int[inputIdsList.size()];
		for(int i=0;i<inputIdsList.size();++i)
		{
			inputIds[i]=inputIdsList.get(i);
		}
		
		this.setInputIdAAndLinkLA(inputIds);
	}
	
	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.lookup.forward();
		this.average.forward();
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		
		if(nextI.length != this.average.output.length )
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		this.average.output = nextI;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer,0);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return this.lookup.getInput(id);
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return this.average.getOutput(id);
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isValid() {
		// TODO Auto-generated method stub
		return true;
	}

}
