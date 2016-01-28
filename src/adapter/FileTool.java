package adapter;

import java.io.BufferedReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class FileTool {
	public static BufferedReader getBufferedReaderFromFile(String filePath,String encoding) throws UnsupportedEncodingException, FileNotFoundException{
		return new BufferedReader(new InputStreamReader(new FileInputStream(filePath),encoding));
	}
	public static BufferedReader getBufferedReaderFromFile(String filePath) throws UnsupportedEncodingException, FileNotFoundException{
		return getBufferedReaderFromFile(filePath,"utf-8");
	}
	public static PrintWriter getPrintWriterForFile(String filePath,boolean isAppend,String encoding) throws UnsupportedEncodingException, FileNotFoundException{
		guaranteeFileDirExist(filePath);
		return new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filePath,isAppend), encoding)));
	}
	public static PrintWriter getPrintWriterForFile(String filePath,String encoding) throws UnsupportedEncodingException, FileNotFoundException{
		return getPrintWriterForFile(filePath,false,encoding);
	}
	public static PrintWriter getPrintWriterForFile(String filePath,boolean isAppend) throws UnsupportedEncodingException, FileNotFoundException{
		return getPrintWriterForFile(filePath,isAppend,"utf-8");
	}
	public static PrintWriter getPrintWriterForFile(String filePath) throws UnsupportedEncodingException, FileNotFoundException{
		return getPrintWriterForFile(filePath,false,"utf-8");
	}
	public static void guaranteeDirExist(String dirPath){
		File dir=new File(dirPath);
		if(!dir.exists()) dir.mkdirs();
	}
	public static void guaranteeFileDirExist(String file){
		File f=new File(file);
		guaranteeDirExist(f.getParent());
	}
	public static void glanceFileContent(String filePath,int start,int end) throws IOException{
		BufferedReader br=new BufferedReader(new FileReader(filePath));
		String curLine="";
		int lineNum=0;
		while((curLine=br.readLine())!=null){
			++lineNum;
			if(lineNum>=start&&lineNum<=end) System.out.println(curLine);
			if(lineNum>end) break;
		}
	}
	public static int getFileLineNum(String filePath) throws Exception{
		return getFileLineNum(new File(filePath));
	}
	public static int getFileLineNum(File file) throws IOException{
		BufferedReader br=new BufferedReader(new FileReader(file));

		String curLine="";
		int lineNum=0;
		while((curLine=br.readLine())!=null){
			++lineNum;
		}
		return lineNum;
	}

	public static String getParentPath(String filePath){
		return getParentPath(new File(filePath));
	}
	public static String getParentPath(File file){
		return file.getParent();
	}
	public static String getFileName(String filePath){
		File f=new File(filePath);
		return f.getName();
	}
	public static String getFileFomatType(String filePath){
		File f=new File(filePath);
		return f.getName().split("\\.")[1];
	}
	public static String getPureFileName(String filePath){
		File f=new File(filePath);
		return f.getName().split("\\.")[0];
	}
	public static String getPureName(String path){
		File f=new File(path);
		if(f.isDirectory()) return f.getName();
		else return getPureFileName(path);
	}
	public static String[] getDirNodes(String file){
		File f=new File(file);
		String dirPath="";
		if(f.isDirectory()||file.charAt(file.length()-1)=='\\'){
			dirPath=f.getAbsolutePath();
		}else{
			dirPath=f.getParent();
		}
		String[] dirNodes=dirPath.split("\\\\");
		return dirNodes;
	}
	public static List<String> getDirNodesList(String file){
		List<String> nodesList=new ArrayList<String>();
		String[] nodes=getDirNodes(file);
		for(String n:nodes) nodesList.add(n);
		return nodesList;
	}
	public static String backReplaceDirNode(String file,String replaceNode,String newNode){
		String[] nodes=getDirNodes(file);
		for(int i=nodes.length-1;i>=0;--i){
			if(nodes[i].equals(replaceNode)) {
				nodes[i]=newNode;
				break;
			}
		}
		StringBuilder stb=new StringBuilder();
		for(String s:nodes){
			stb.append(s+"\\");
		}
		return stb.toString();
	}
	public static String forwardReplaceDirNode(String file,String replaceNode,String newNode){
		String[] nodes=getDirNodes(file);
		for(int i=0;i<nodes.length;++i){
			if(nodes[i].equals(replaceNode)) {
				nodes[i]=newNode;
				break;
			}
		}
		StringBuilder stb=new StringBuilder();
		for(String s:nodes){
			stb.append(s+"\\\\");
		}
		return stb.toString();
	}
	
	public static String forwardInsertDirNode(String file,String insertPointNode,String newNode){
		String[] nodes=getDirNodes(file);
		int insertPoint=0;
		for(int i=nodes.length-1;i>=0;--i ){
			if(nodes[i].equals(insertPointNode)){
				insertPoint=i;
				break;
			}
		}
		StringBuilder stb=new StringBuilder();
		for(int i=0;i<insertPoint;++i){
			stb.append(nodes[i]+"\\\\");
		}
		stb.append(newNode+"\\\\");
		for(int i=insertPoint;i<nodes.length;++i){
			stb.append(nodes[i]+"\\\\");
		}
		return stb.toString();

	}
	//�ӵ�һ���ڵ������һ���ڵ�������ҵ���һ�������ڵ�ʱ���ڸò���ڵ�֮������½ڵ�
	public static String backInsertDirNode(String file,String insertPointNode,String newNode){
		String[] nodes=getDirNodes(file);
		int insertPoint=0;
		for(int i=0;i<nodes.length;++i ){
			if(nodes[i].equals(insertPointNode)){
				insertPoint=i;
				break;
			}
		}
		StringBuilder stb=new StringBuilder();
		for(int i=0;i<=insertPoint;++i){
			stb.append(nodes[i]+"\\\\");
		}
		stb.append(newNode+"\\\\");
		for(int i=insertPoint+1;i<nodes.length;++i){
			stb.append(nodes[i]+"\\\\");
		}
		return stb.toString();

	}
	public static void copy(String srcFile,int startLineNum,int endLineNum,String destFile) throws IOException{
		copy(new File(srcFile),startLineNum,endLineNum,new File(destFile));
	}

	public static void copy(File srcFile,int startLineNum,int endLineNum,File destFile) throws IOException{
		BufferedReader reader=FileTool.getBufferedReaderFromFile(srcFile.getAbsolutePath());
		PrintWriter writer=FileTool.getPrintWriterForFile(destFile.getAbsolutePath());
		int lineNum=1;
		String line="";
		while((line=reader.readLine())!=null){
			if(lineNum>=startLineNum){
				if(lineNum>endLineNum){break;} 
				writer.write(line);
				writer.write("\r\n");
			}
			++lineNum;
		}
		writer.close();
	}

	public static void copy(String srcFile,String destFile) throws IOException, Exception{
		copy(srcFile,1,FileTool.getFileLineNum(srcFile),destFile);
	}
	public static String getGrandParentPath(String filePath){
		return getParentPath(getParentPath(filePath));
	}

}
