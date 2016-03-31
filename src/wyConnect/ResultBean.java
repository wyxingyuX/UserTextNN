package wyConnect;

public class ResultBean {
	public PredictResult preTest_accVal;
	public int MaxAccValidationRound;

	public PredictResult preTest_meanErrorVal;
	public int minErrorValidationRound;

	public PredictResult maxAccTest;
	public int maxAccRound;
	
	public PredictResult fitTrainA;
	public PredictResult fitTrainM;

	public ResultBean(){}

	public void setPreTestByAccVal(PredictResult preTest_accVal,int MaxAccValidationRound)
	{
		this.preTest_accVal=preTest_accVal;
		this.MaxAccValidationRound=MaxAccValidationRound;
	}
	public void setPreTestByMeanErrorVal(PredictResult preTest_meanErrorVal,int minErrorValidationRound)
	{
		this.preTest_meanErrorVal=preTest_meanErrorVal;
		this.minErrorValidationRound=minErrorValidationRound;

	}
	public void setMaxAccTest(PredictResult maxAccTest,int maxAccRound)
	{

		this.maxAccTest=maxAccTest;
		this.maxAccRound=maxAccRound;
	}
}
