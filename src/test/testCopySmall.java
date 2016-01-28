package test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Collections;
import java.util.List;

import adapter.FileTool;
import adapter.WYIO;

public class testCopySmall {
	public static void excCopySmall() throws IOException{
		String dir="F:\\ExpData\\DataFromOther\\dytang-cnn\\yelp-13\\";
		File d=new File(dir);
		File[] fs=d.listFiles();
		for(File f:fs){
			String[] dirNodes=FileTool.getDirNodes(f.getAbsolutePath());
			String dest=FileTool.forwardInsertDirNode(f.getAbsolutePath(),dirNodes[dirNodes.length-1], "Test")+f.getName();
			int end=FileTool.getFileLineNum(f)/100;
			System.out.println(end);
			FileTool.copy(f.getAbsolutePath(), 1,end ,dest);
		}

	}
	
	public static void excRandomSelectSmall() throws IOException{
		String dir="F:\\ExpData\\DataFromOther\\dytang-cnn\\yelp-13\\";
		File d=new File(dir);
		File[] fs=d.listFiles();
		for(File f:fs){
			String[] dirNodes=FileTool.getDirNodes(f.getAbsolutePath());
			String dest=FileTool.forwardInsertDirNode(f.getAbsolutePath(),dirNodes[dirNodes.length-1], "Test")+f.getName();
			List<String> contents=WYIO.readFile2List(f.getAbsolutePath());
			Collections.shuffle(contents);
			int end=contents.size()/100;
			System.out.println(end);
			WYIO.writeList2File(contents.subList(0, end), dest);
			
			
		}
	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		excRandomSelectSmall();
	}

}
