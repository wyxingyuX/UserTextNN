package wyConnect;

public class ResultBean {
  public double acc_MaxAccValidation;
  public int MaxAccValidationRound;
  
  public double acc_minErrorValidation;
  public int minErrorValidationRound;
  
  public double acc_max;
  public int maxAccRound;
  
  public ResultBean(){}
  public ResultBean(double acc_MaxAccValidation, double acc_minErrorValidation){
	  this.acc_MaxAccValidation=acc_MaxAccValidation;
	  this.acc_minErrorValidation=acc_minErrorValidation;
  }
}
