package test;

import java.io.IOException;

import adapter.ToolKit;

public class generateExpSet {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String base="F:\\ExpData\\DataIntegate\\source\\nne\\publicInfo\\Tags\\dyt\\textData\\";
		String testFile=base+"dytText_test.txt";
		String trainFile=base+"dytText_train.txt";
		String destFile=base+"allReview.txt";
//		ToolKit.mergeRerviews(trainFile,testFile,destFile);
		String idCatesFile=base+"ids.txt";
//		ToolKit.extractIdCates(destFile, "\t\t", idCatesFile, "\t");
		int fold=5;
		ToolKit.generaterFoldTrainValidTestExpSet(idCatesFile, "\t", base+fold+"fold\\", fold);
	}

}
