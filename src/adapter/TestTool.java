package adapter;

import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;

public class TestTool {
	public static  void print(Object o){
		System.out.print(o);
		try {
			GlobleLog.write(o.toString());
		} catch (UnsupportedEncodingException | FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static  void println(Object o){
		System.out.println(o);
		try {
			GlobleLog.write(o.toString()+"\r\n");
		} catch (UnsupportedEncodingException | FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
