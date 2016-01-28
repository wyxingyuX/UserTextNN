package wyConnect;

import java.util.Random;

import nnet.LinearLayer;
import nnet.NNInterface;
import nnet.TanhLayer;

public class TanhLinear implements NNInterface {
	
	public int inputLength;
	public int outputLength;
	public LinearLayer linear;
	public TanhLayer tanh;
	
	int linkId;
	
	public TanhLinear(){

	}

	public TanhLinear(int xInputLength,int xOutputLength) throws Exception{
		inputLength=xInputLength;
		outputLength = xOutputLength;
		
		tanh = new TanhLayer(inputLength);
		linear = new LinearLayer(inputLength, outputLength);
		
		tanh.link(linear);
	}

	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear.randomize(r, min, max);
		tanh.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		tanh.forward();
		linear.forward();
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		linear.backward();
		tanh.backward();
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		linear.update(learningRate);
		tanh.update(learningRate);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		linear.updateAdaGrad(learningRate, batchsize);
		tanh.updateAdaGrad(learningRate, batchsize);
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		tanh.clearGrad();
		linear.clearGrad();
		
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != linear.output.length || nextIG.length != linear.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		linear.output = nextI;
		linear.outputG = nextIG;
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		 link(nextLayer,linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return tanh.getInput(id);
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return linear.getOutput(id);
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return tanh.getInputG(id);
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return linear.getOutputG(id);
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		LinearTanh clone = new LinearTanh();

		clone.inputLength=inputLength;
		clone.outputLength = outputLength;
		clone.linkId=linkId;
		
		clone.linear = (LinearLayer)linear.cloneWithTiedParams();
		clone.tanh = (TanhLayer)tanh.cloneWithTiedParams();

		try {
			clone.tanh.link(clone.linear);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return clone;
	}

	@Override
	public boolean isValid() {
		// TODO Auto-generated method stub
		return true;
	}

}
