package test;

import java.io.IOException;

import adapter.ToolKit;

public class testExtractReview {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String dir="F:\\ExpData\\DataFromOther\\dytang-cnn\\Test\\";
		ToolKit.prepareDataForWordEmbeding(dir+"yelp-13\\", dir+"reviews.txt");
	}

}
