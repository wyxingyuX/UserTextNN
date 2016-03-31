package wyConnect;

import java.util.ArrayList;
import java.util.List;

public class PredictResult {
   public double acc;
   public double meanError;
   public double macroF;
   public List<PredictData> predictDatas=new ArrayList<PredictData>();
  
   public PredictResult()
   {  
	   this.meanError=Double.MAX_VALUE;
	   this.acc=Double.MIN_VALUE;
	   this.macroF=Double.MIN_VALUE;
   }
   public PredictResult(double acc,double meanError){
	   this.acc=acc;
	   this.meanError=meanError;
   }
   public PredictResult(double acc,double meanError,double macroF,List<PredictData> predictDatas)
   {
	   this.acc=acc;
	   this.meanError=meanError;
	   this.macroF=macroF;
	   this.predictDatas=predictDatas;
   }
   
   public void add(PredictData pdata)
   {
	   predictDatas.add(pdata);
   }
}
