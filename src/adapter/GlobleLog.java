package adapter;

import java.io.FileNotFoundException;

import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;

import wyConnect.View;

public class GlobleLog {
	private static Log log;
	
//	public static void parseLogPath(List<View> views) throws UnsupportedEncodingException, FileNotFoundException{
//		if(views.size()==1){
//			setLogDir(FileTool.getGrandParentPath(views.get(0).allDatas)+"\\");
//		}
//		if(views.size()>1){
//			String dirNodeName="";
//			for(int i=0;i<views.size();++i){
//				if(i==views.size()-1){
//					dirNodeName=views.get(0).viewName;
//				}
//				dirNodeName=views.get(0).viewName+"+";
//			}
//			String base=FileTool.getGrandParentPath(FileTool.getParentPath(views.get(0).allDatas))+"\\";
//			setLogDir(base+dirNodeName+"\\");
//		}
//	}

	public static void setLogDir(String logDir) throws UnsupportedEncodingException, FileNotFoundException {
		log=null;
		log=new Log(logDir);
	}

	public static void write(String s) throws UnsupportedEncodingException, FileNotFoundException{
		if(null!=log)
			log.write(s);
	}

}
