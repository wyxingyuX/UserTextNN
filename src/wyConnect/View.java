package wyConnect;

public class View {
	public String embeddingFileWord;
	public int embeddingLengthWord;
	public String trainIds;
	public String testIds;
	public String validationIds;
	public String allDatas;
	public String viewName;
	public boolean isCNN=true;

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
			boolean isCNN,
			String embeddingFileWord,
			int embeddingLengthWord,
			String trainIds,
			String validationIds,
			String testIds,
			String allDatas){
		this.viewName=viewName;
		this.isCNN=isCNN;
		this.embeddingFileWord=embeddingFileWord;
		this.embeddingLengthWord=embeddingLengthWord;
		this.trainIds=trainIds;
		this.validationIds=validationIds;
		this.testIds=testIds;
		this.allDatas=allDatas;

	}
}
