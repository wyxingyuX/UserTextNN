package adapter;

import java.io.FileNotFoundException;

import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.SimpleDateFormat;
import java.util.Date;
public class Log {
	private String logPath;
	public Log(String dir) throws UnsupportedEncodingException, FileNotFoundException{
		this.logPath=dir+"log.txt";
		this.writeDate();
	}
	public void write(String s) throws UnsupportedEncodingException, FileNotFoundException{
		PrintWriter pw=FileTool.getPrintWriterForFile(logPath, true);
		pw.write(s);
		pw.close();
	}
	protected void writeDate() throws UnsupportedEncodingException, FileNotFoundException{
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		this.write("------------------------------------"+df.format(new Date())+"------------------------------------------\r\n");
	}

}
