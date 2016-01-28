package adapter;

import java.io.BufferedReader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import nnet.AverageLayer;
import wyConnect.CNN4ViewUser;

public  class ToolKit {
	public static void prepareDataForWordEmbeding(String dir,String dest) throws IOException{
		File d=new File(dir);
		File[] fs=d.listFiles();
		List<String> reviewList=new LinkedList<String>();

		for(File f:fs){
			BufferedReader reader=FileTool.getBufferedReaderFromFile(f.getAbsolutePath());
			String line="";
			while((line=reader.readLine())!=null){
				String[] elms=line.split("\t\t");
				String filterStr=elms[3].replace("<sssss>", "");
				reviewList.add(filterStr);
			}
			reader.close();
		}

		WYIO.write(reviewList, dest);
	}
	public static void printDiem(String filePath) throws IOException{
		BufferedReader reader=FileTool.getBufferedReaderFromFile(filePath);
		String line=reader.readLine();
		String[] elms=line.split("\\s{1,}");
		System.out.println("diem is "+(elms.length-1));
	}

	public static void writeDocsAverage2File(CNN4ViewUser net4User,String destFile,boolean isAppend) throws UnsupportedEncodingException, FileNotFoundException{
		PrintWriter writer=FileTool.getPrintWriterForFile(destFile,isAppend);
		String seperater="\t";

		writer.write(net4User.getData().userStr);
		AverageLayer average=net4User.average;
		for(int i=0;i<average.output.length;++i){
			writer.write(seperater+average.output[i]);
		}
		writer.write("\r\n");
		writer.close();
	}

	public static void mergeRerviews(String ...files) throws IOException{
		PrintWriter writer=FileTool.getPrintWriterForFile(files[files.length-1]);
		for(int i=0;i<files.length-1;++i){
			String sourceFile=files[i];
			BufferedReader reader=FileTool.getBufferedReaderFromFile(sourceFile);
			String line="";
			while((line=reader.readLine())!=null){
				writer.write(line+"\r\n");
			}
		}
		writer.close();
	}

	public static void extractIdCates(String reviewsFile,String reviewSeperater
			,String idCatesFile,String idCateSeperater) throws IOException{
		BufferedReader reader=FileTool.getBufferedReaderFromFile(reviewsFile);
		PrintWriter writer=FileTool.getPrintWriterForFile(idCatesFile);
		String line="";
		while((line=reader.readLine())!=null){
			String[] elms=line.split(reviewSeperater);
			String id=elms[0].trim();
			String cate=elms[2].trim();
			writer.write(id+idCateSeperater+cate+"\r\n");
		}
		writer.close();
	}

	public static void generateFoldTrainTestExpSet(String idCatesFile,String idsCateSeperater,
			String destDir,int fold) throws IOException{
		Map<String ,List<String>> cateIdsMap=new HashMap<String,List<String>>();
		BufferedReader reader=FileTool.getBufferedReaderFromFile(idCatesFile);
		String line="";
		while((line=reader.readLine())!=null){
			String[] elms=line.split(idsCateSeperater);
			String id=elms[0].trim();
			String cate=elms[1].trim();
			List<String> ids=cateIdsMap.get(cate);
			if(ids==null){
				ids=new ArrayList<String>();
				cateIdsMap.put(cate, ids);
			}
			ids.add(id);
		}
		for(Map.Entry<String, List<String>> entry:cateIdsMap.entrySet()){
			Collections.sort(entry.getValue());
		}

		for(int i=0;i<fold;++i){
			PrintWriter trainWriter=FileTool.getPrintWriterForFile(destDir+i+"\\train.txt");
			PrintWriter testWriter=FileTool.getPrintWriterForFile(destDir+i+"\\test.txt");
			for(Map.Entry<String, List<String>> entry:cateIdsMap.entrySet()){
				String cate=entry.getKey();
				List<String> ids=entry.getValue();
				int foldNum=ids.size()/fold;

				for(int k=0;k<ids.size();++k){
					if(k>=i*foldNum&&k<(i+1)*foldNum){
						testWriter.write(ids.get(k)+idsCateSeperater+cate+"\r\n");
					}else{
						trainWriter.write(ids.get(k)+idsCateSeperater+cate+"\r\n");
					}
				}
			}
			trainWriter.close();
			testWriter.close();
		}
	}

	public static void generaterFoldTrainValidTestExpSet(String idCatesFile,String idsCateSeperater,
			String destDir,int fold) throws IOException{
		Map<String ,List<String>> cateIdsMap=new HashMap<String,List<String>>();
		BufferedReader reader=FileTool.getBufferedReaderFromFile(idCatesFile);
		String line="";
		while((line=reader.readLine())!=null){
			String[] elms=line.split(idsCateSeperater);
			String id=elms[0].trim();
			String cate=elms[1].trim();
			List<String> ids=cateIdsMap.get(cate);
			if(ids==null){
				ids=new ArrayList<String>();
				cateIdsMap.put(cate, ids);
			}
			ids.add(id);
		}
		for(Map.Entry<String, List<String>> entry:cateIdsMap.entrySet()){
			Collections.sort(entry.getValue());
		}

		for(int i=0;i<fold;++i){
			PrintWriter trainWriter=FileTool.getPrintWriterForFile(destDir+i+"\\train.txt");
			PrintWriter validationWriter=FileTool.getPrintWriterForFile(destDir+i+"\\validation.txt");
			PrintWriter testWriter=FileTool.getPrintWriterForFile(destDir+i+"\\test.txt");
			for(Map.Entry<String, List<String>> entry:cateIdsMap.entrySet()){
				String cate=entry.getKey();
				List<String> ids=entry.getValue();
				int foldNum=ids.size()/fold;

				for(int k=0;k<ids.size();++k){
					if(k>=i*foldNum&&k<(i+1)*foldNum){
						testWriter.write(ids.get(k)+idsCateSeperater+cate+"\r\n");
					}else if(k>=((i+1)%fold)*foldNum&&k<((i+1)%fold+1)*foldNum){
						validationWriter.write(ids.get(k)+idsCateSeperater+cate+"\r\n");
					}else{
						trainWriter.write(ids.get(k)+idsCateSeperater+cate+"\r\n");
					}
				}
			}
			trainWriter.close();
			validationWriter.close();
			testWriter.close();
		}
	}

	public static Map.Entry<Integer,Double> getMaxValue(List<Double> arrays){
		Double maxValue=Double.MIN_VALUE;
		int index=-1;
		for(int i=0;i<arrays.size();++i){
			if(arrays.get(i)>maxValue){
				maxValue=arrays.get(i);
				index=i;
			}
		}
		HashMap<Integer,Double> map=new HashMap<Integer,Double>();
		map.put(index, maxValue);
		Map.Entry<Integer, Double> entry=null;
		for(Map.Entry<Integer, Double> en:map.entrySet()){
			entry=en;
		}
		return entry ;
	}
	public static Map.Entry<Integer,Double> getMinValue(List<Double> arrays){
		Double minValue=Double.MAX_VALUE;
		int index=-1;
		for(int i=0;i<arrays.size();++i){
			if(arrays.get(i)<minValue){
				minValue=arrays.get(i);
				index=i;
			}
		}
		HashMap<Integer,Double> map=new HashMap<Integer,Double>();
		map.put(index, minValue);
		Map.Entry<Integer, Double> entry=null;
		for(Map.Entry<Integer, Double> en:map.entrySet()){
			entry=en;
		}
		return entry ;
	}

	public static Map.Entry<Integer, Double> getMinimalPoint(List<Double> arrays,double threshold){
		Double delataMeanError=Double.MIN_VALUE;
		int index=-1;
		boolean lastIsDsc=true;
		for(int i=0;i<arrays.size();++i){
			Double meanError=arrays.get(i);
			if(meanError>0&&lastIsDsc&&meanError>threshold){
				lastIsDsc=false;
				delataMeanError=meanError;
				index=i;
				break;
			}else if(meanError>0&&!lastIsDsc){
				lastIsDsc=false;
				delataMeanError=arrays.get(i-1);
				index=i-1;
				break;
			}else if(meanError>0&&lastIsDsc&&meanError<=threshold){
				lastIsDsc=false;
			}else{
				lastIsDsc=true;
			}
		}

		HashMap<Integer,Double> map=new HashMap<Integer,Double>();
		map.put(index, delataMeanError);
		Map.Entry<Integer, Double> entry=null;
		for(Map.Entry<Integer, Double> en:map.entrySet()){
			entry=en;
		}
		return entry ;
	}

}
