package wyConnect;

import java.util.List;

public class ViewInfo {
	public String viewName;
	public String embedingFile;
	public int embedingLength;
	public String allDataView;
	public double viewWeight;//different view can set different weight
	public int netType=2;// 0 LR, 1 CNN, 2 CNNLR
	public boolean isUseCommonFilters=false;//is this view use common cnn filter? 
	public int viewOutputLength;
	
	public List<Integer> windowSizeLookupLst;

	public ViewInfo(){
	}
	public ViewInfo(String viewName,String embedingFile,int embedingLength,String allDataView){
		this( viewName, embedingFile, embedingLength,allDataView,1.0);
	}
	public ViewInfo(String viewName,String embedingFile,int embedingLength,String allDataView,int netType){
		this( viewName, embedingFile, embedingLength,allDataView,1.0);
		this.netType=netType;
	}
	public ViewInfo(String viewName,String embedingFile,int embedingLength,int outputLength,List<Integer> wsizeLookupLst,String allDataView,int netType,boolean isUseCommonFilters){
		this( viewName, embedingFile, embedingLength,allDataView,1.0);
		this.netType=netType;
		this.isUseCommonFilters=isUseCommonFilters;
		this.viewOutputLength=outputLength;
		this.windowSizeLookupLst=wsizeLookupLst;
	}
	public ViewInfo(String viewName,String embedingFile,int embedingLength,String allDataView,double viewWeight){
		this.viewName=viewName;
		this.embedingFile=embedingFile;
		this.embedingLength=embedingLength;
		this.allDataView=allDataView;
		this.viewWeight=viewWeight;
	}


}
