package wyConnect;

import java.util.Random;

import nnet.LinearLayer;
import nnet.LookupLayer;
import nnet.LookupLinearTanh;
import nnet.NNInterface;
import nnet.TanhLayer;

public class LinearTanh implements NNInterface {
	public int inputLength;
	public int outputLength;
	public LinearLayer linear;
	public TanhLayer tanh;
	
	int linkId;
	
	public LinearTanh(){

	}
	public LinearTanh(LinearLayer seedLinear) throws Exception{
		inputLength=seedLinear.inputLength;
		outputLength = seedLinear.outputLength;
		linear = (LinearLayer) seedLinear.cloneWithTiedParams();
		tanh = new TanhLayer(outputLength);

		linear.link(tanh);
	}

	public LinearTanh(int xInputLength,int xOutputLength) throws Exception{
		inputLength=xInputLength;
		outputLength = xOutputLength;

		linear = new LinearLayer(inputLength, outputLength);
		tanh = new TanhLayer(outputLength);

		linear.link(tanh);
	}
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear.randomize(r, min, max);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		linear.forward();
		tanh.forward();
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		tanh.backward();
		linear.backward();
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		tanh.update(learningRate);
		linear.update(learningRate);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		tanh.updateAdaGrad(learningRate, batchsize);
		linear.updateAdaGrad(learningRate, batchsize);

	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		linear.clearGrad();
		tanh.clearGrad();

	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != tanh.output.length || nextIG.length != tanh.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		tanh.output = nextI;
		tanh.outputG = nextIG;

	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
        link(nextLayer,linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return linear.getInput(id);
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return tanh.getOutput(id);
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return linear.getInputG(id);
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return tanh.getOutputG(id);
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
			clone.linear.link(clone.tanh);
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
	@Override
	public void update(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

}
