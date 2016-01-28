package adapter;

import java.util.ArrayList;

import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

public class CollectionTool {
	public static List<String> popSubList(List<String> list,int fromIndex,int toIndex){
		List<String> subList=new LinkedList<String>();
		int count=0;
		while(toIndex>fromIndex&&count<(toIndex-fromIndex)){
			subList.add(list.get(fromIndex));
			list.remove(fromIndex);
			++count;
		}
		return subList;
	}
	public static List<String> mergeStrList(List<String> list1,List<String> list2){
		List<String> list=new ArrayList<String>();
		for(String s: list1) list.add(s);
		for(String s: list2) list.add(s);
		return list;
	}
	public static List<Object> mergeList(List<Object> list1,List<Object> list2){
		List<Object> list=new LinkedList<Object>();
		for(Object o:list1) list.add(o);
		for(Object o:list2) list.add(o);
		return list;
	}
	public static List<String> toLinkedList(String[] t){
		LinkedList<String> list=new LinkedList<String>();
		for(String s:t) list.add(s);
		return list;
	}
	public static String[] set2Array(Set<String> set){
		String[] array=new String[set.size()];
		int i=0;
		for(String s:set) {
			array[i]=s;
			++i;
		}
		return array;
	}
   public static Set<String> trimDuplicate(List<String> lists){
	   Set set=new LinkedHashSet<String>();
	   for(String s:lists){
		   set.add(s);
	   }
	   return set;
   }
}
