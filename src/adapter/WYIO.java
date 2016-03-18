package adapter;

import java.io.BufferedReader;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import other.Data;

public class WYIO {
	public static void write(List<String> lists,String dest) throws UnsupportedEncodingException, FileNotFoundException{
		PrintWriter writer=FileTool.getPrintWriterForFile(dest);
		for(String s:lists){
			writer.write(s);
			writer.write("\r\n");
		}
		writer.close();
	}
	public static List<String> readFile2List(String filePath) throws IOException{
		BufferedReader reader=FileTool.getBufferedReaderFromFile(filePath);
		List<String> list=new ArrayList<String>();
		String line="";
		while((line=reader.readLine())!=null){
			list.add(line);
		}
		reader.close();
		return list;
	}
	public static void writeList2File(List<String> list,String dest ) throws UnsupportedEncodingException, FileNotFoundException{
		PrintWriter writer=FileTool.getPrintWriterForFile(dest);
		for(String s:list){
			writer.write(s);
			writer.write("\r\n");
		}
		writer.close();
	}
	public static void readCorpus(String ids,String idsSeperater,String allWords,String allWordsSeperater,List<Data> datas) throws IOException{
		BufferedReader idsReader=FileTool.getBufferedReaderFromFile(ids);
		BufferedReader allWordsReader=FileTool.getBufferedReaderFromFile(allWords);
		Map<String,String> idsMap=new LinkedHashMap<String,String>();
		String idsLine="";
		while((idsLine=idsReader.readLine())!=null){
			String[] elms=idsLine.split(idsSeperater);
			idsMap.put(elms[0].trim(), elms[0].trim());
		}
		
		String allWordsLine="";
		while((allWordsLine=allWordsReader.readLine())!=null){
			String [] elms=allWordsLine.split(allWordsSeperater);
			String id=elms[0].trim();
			String source=elms[1];
			int cate=Integer.parseInt(elms[2]);
			String words=elms[3];
			if(idsMap.containsKey(id)){
				datas.add(new Data(id,source,words,cate));
			}
		}

	}
	public  static int readVecDiem(String wordVecFile,String seperater,int lineNum) throws IOException
	{
		BufferedReader reader=FileTool.getBufferedReaderFromFile(wordVecFile);
		String line="";
		int count=0;
		while((line=reader.readLine())!=null)
		{
			++count;
			if(lineNum==count)
			{
				break;
			}
		}
		String[] elms=line.split(seperater);
		System.out.println(lineNum+" line vec diem is "+(elms.length-1));
		return elms.length-1;
	}
	public static int readVecDiem(String wordVecFile,String seperater) throws IOException
	{
		return readVecDiem(wordVecFile,seperater,2);
	
 	}
}
