package com.bham.pij.assignments.calculator;
//Thomas Armstrong 2179997 
import java.util.ArrayList;

public class Stack 
{
	ArrayList<String> stackArray = new ArrayList<String>();
	
	public String pop()
	{
		String value = stackArray.get(0);
		stackArray.remove(0);
		return value;		
	}
	
	public void push(String expression)
	{
		stackArray.add(0, expression);
	}
	
	public void push(float expression)
	{
		String stringExpression = Float.toString(expression);
		push(stringExpression);
	}
	
	public boolean isEmpty()
	{
		return stackArray.isEmpty();
	}
	
	public int size()
	{
		return stackArray.size();
	}
}
