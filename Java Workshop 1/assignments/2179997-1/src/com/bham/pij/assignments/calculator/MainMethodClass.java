package com.bham.pij.assignments.calculator;
//Thomas Armstrong 2179997 

import java.util.Scanner;
public class MainMethodClass {

	public static void main(String[] args) 
	{
		Scanner sc = new Scanner(System.in);
		Calculator myCalculator = new Calculator();
		String input = " ";
		System.out.println("Enter x to exit at any time.");
		while  (!input.equals("x"))
		{
			input = sc.nextLine();
			if ((!input.equals("")) && !(input.equals("x")))
				myCalculator.functionSelection(input.trim());
			
			
		}
	}
	
	

}
