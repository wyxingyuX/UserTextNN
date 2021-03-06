package wyConnect;

import java.util.List;

public class View {
	public String embeddingFileWord;
	public int embeddingLengthWord;
	public String trainIds;
	public String testIds;
	public String validationIds;
	public String allDatas;
	public String viewName;
	public int netType=2;// 0 LR, 1 cnn, 2 cnnlr
	public double viewWeight=1.0;//different view can set different weight
	public boolean isUseCommonFiters=false;
	
	public int viewOutputLength;
	public List<Integer> windowSizeLookupLst;
   
	public View(ViewInfo vinfo,
			String trainIds,
			String validationIds,
			String testIds){
		this(vinfo.viewName,vinfo.embedingFile,vinfo.embedingLength,trainIds,validationIds,testIds,vinfo.allDataView,vinfo.viewWeight);
		
	}
	public View(
			ViewInfo vinfo,
			int netType,
			String trainIds,
			String validationIds,
			String testIds){
		this(vinfo.viewName,vinfo.embedingFile,vinfo.embedingLength,trainIds,validationIds,testIds,vinfo.allDataView,vinfo.viewWeight);
		this.netType=netType;
	}
	public View(
			ViewInfo vinfo,
			int netType,
			boolean isUseCommonFilters,
			String trainIds,
			String validationIds,
			String testIds){
		this(vinfo.viewName,vinfo.embedingFile,vinfo.embedingLength,trainIds,validationIds,testIds,vinfo.allDataView,vinfo.viewWeight);
		this.netType=netType;
		this.isUseCommonFiters=isUseCommonFilters;
		this.viewOutputLength=vinfo.viewOutputLength;
		this.windowSizeLookupLst=vinfo.windowSizeLookupLst;
	}

	public View(String viewName,
			String embeddingFileWord,
			int embeddingLengthWord,
			String trainIds,
			String validationIds,
			String testIds,
			String allDatas){
		this.viewName=viewName;
		this.embeddingFileWord=embeddingFileWord;
		this.embeddingLengthWord=embeddingLengthWord;
		this.trainIds=trainIds;
		this.validationIds=validationIds;
		this.testIds=testIds;
		this.allDatas=allDatas;
	}
	public View(String viewName,
			String embeddingFileWord,
			int embeddingLengthWord,
			String trainIds,
			String validationIds,
			String testIds,
			String allDatas,
			double viewWeight){
		this.viewName=viewName;
		this.embeddingFileWord=embeddingFileWord;
		this.embeddingLengthWord=embeddingLengthWord;
		this.trainIds=trainIds;
		this.validationIds=validationIds;
		this.testIds=testIds;
		this.allDatas=allDatas;
		this.viewWeight=viewWeight;
	}
	
	public View(String viewName,
			int netType,
			String embeddingFileWord,
			int embeddingLengthWord,
			String trainIds,
			String validationIds,
			String testIds,
			String allDatas){
		this.viewName=viewName;
		this.netType=netType;
		this.embeddingFileWord=embeddingFileWord;
		this.embeddingLengthWord=embeddingLengthWord;
		this.trainIds=trainIds;
		this.validationIds=validationIds;
		this.testIds=testIds;
		this.allDatas=allDatas;

	}
	
	public String getWsizeStr()
	{
		StringBuilder stb=new StringBuilder();
		for(int w:this.windowSizeLookupLst)
		{
			stb.append(w+"");
		}
		return stb.toString();
	}
}
