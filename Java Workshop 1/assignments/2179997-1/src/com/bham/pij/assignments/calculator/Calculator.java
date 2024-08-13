package com.bham.pij.assignments.calculator;
//Thomas Armstrong 2179997 
import java.util.ArrayList;
/**
 * 
 * @author tom
 * @version 1.0
 * @date 27/10/2020
 * @description Assignment 2, the basic calculator class
 */

/**
 * 
 * @author tom
 * @param expression: String expression in the form a + b or with
 *  the other operators
 */
public class Calculator {
	
	private float currentValue;
	
	private float memoryValue;
	
	private ArrayList<Float> historyArrayList = new ArrayList<Float>();			
	
	private Stack stack = new Stack();
	
	/**
	 * default constructor
	 */
	public Calculator()
	{
		
	}

	public float evaluate(String expression)
	{
		float result = Float.MIN_VALUE;
		if (expression.contains(" "))
		{
		String[] splitExpression = expression.split(" ");
		
			if (isValidBasicExpression(expression))
			{
				result = evaluateExpression(splitExpression[0], splitExpression[1], splitExpression[2]);
				setCurrentValue(result);
			}
			else if (isMemoryExpression(expression))
			{
				result = evaluateExpression(getMemoryValue(), splitExpression[0], splitExpression[1]);
				setCurrentValue(result);
			}
			else if (isBracketedExpression(expression))
			{
				String operator = splitExpression[3];
				String[] brackets = splitBrackets(expression);
				brackets[0] = removeBrackets(brackets[0]);
				String[] splitBracket1 = brackets[0].split(" ");
				brackets[1] = removeBrackets(brackets[1]);
				String[] splitBracket2 = brackets[1].split(" ");
				
				result = evaluateExpression(evaluateExpression(splitBracket1[0], splitBracket1[1], splitBracket1[2]), 
						operator,
						evaluateExpression(splitBracket2[0], splitBracket2[1], splitBracket2[2]));
				
				setCurrentValue(result);
			}
			else if (isLongExpression(expression))
			{
				result = solveLongExpression(expression);
				setCurrentValue(result);
			}
			else
			{
				System.out.println("Invalid Input");
				setCurrentValue(0);
			}
				
		}
		else
		{
			System.out.println("Invalid Input");
			setCurrentValue(0);
		}
		if (getCurrentValue() != 0)
			addToHistory();
		return result;
		
	}
	
	
	
	/**
	 * 
	 * @param operand
	 * @return whether or not operand  is a valid operand 
	 */
	
	public boolean isValidOperand(String operand)
	{
		try
		{
			Double.parseDouble(operand);
			return true;
		} 
		catch(NumberFormatException e)
		{
			return false;
		}
	}
	
	public boolean isValidOperator(String operator)
	{
		if (operator.length() != 1)
		{
			return false;
		}
		else
		{
			switch (operator.charAt(0))
			{
				case '+':
				{
					return true;
				}
				case '-':
				{
					return true;
				}
				case '*':
				{
					return true;
				}
				case '/':
				{
					return true;
				}
				default:
				{
					return false;
				}
			}
		}
	}
	
	public boolean isValidBasicExpression(String expression)
	{
		String[] splitExpression = expression.split(" ");
		boolean result = false;
		if (splitExpression.length == 3)
		{
			if ((isValidOperand(splitExpression[0])) && (isValidOperand(splitExpression[2])))
			{
				if (isValidOperator(splitExpression[1]))
				{
					if (!(isDivideByZeroError(splitExpression[1], splitExpression[2])))
					{
						result = true;
					}
					
				}
			}
		}
		return result;
	}
	
	public boolean isMemoryExpression(String expression)
	{
		String[] splitExpression = expression.split(" ");
		boolean result = false;
		if (splitExpression.length == 2)
		{
			if (isValidOperand(splitExpression[1]))
			{
				if (isValidOperator(splitExpression[0]))
				{
					if (!(isDivideByZeroError(splitExpression[0], splitExpression[1])))
					{
						result = true;
					}
					
				}
			}
		}
		
		return result;
	}
	
	/**
	 * 
	 * @param Expression
	 * @return if there is a 0 being divided.
	 */
	public boolean isDivideByZeroError(String operator, String operand)
	{
		return ((operator.equals("/")) && (operand.equals("0")));
	}

	public float getCurrentValue() 
	{
		return currentValue;
	}

	public float getMemoryValue() 
	{
		return memoryValue;
	}

	public void setMemoryValue(float memval) 
	{
		memoryValue = memval;
	}
	
	public void clearMemory() 
	{
		memoryValue = 0;
	}
	
	public void functionSelection(String userInput)
	{
		if ((userInput.trim()).equals("m"))
		{
			//storing as memory
			setMemoryValue(getCurrentValue()); 
		}
		else if ((userInput.trim()).equals("mr"))
		{
			System.out.println(getMemoryValue());
		}
		else if ((userInput.trim()).equals("c"))
		{
			clearMemory();
		}
		else if ((userInput.trim().equals("h")))
		{
			displayHistory();
		}
		else
			System.out.println(evaluate(userInput));	
	}

	public void setCurrentValue(float currentValue) 
	{
		this.currentValue = currentValue;
	}
	
	public float getHistoryValue(int index)
	{
		return historyArrayList.get(index);
	}
	
	public void addToHistory()
	{
		historyArrayList.add(getCurrentValue());
	}
	
	public void displayHistory()
	{
		for (int i = 0; i < historyArrayList.size(); i ++)
		{
			if (i != historyArrayList.size() - 1)
				System.out.print(historyArrayList.get(i) + ", ");
			else
				System.out.println(historyArrayList.get(i));
		}
	}
	
	public boolean isBracketed(String expression)
	{
		boolean result = false;
		if  (expression.length() > 0)
		{
			if ((expression.charAt(0) == '(') && (expression.charAt(expression.length() - 1) == ')'))
				{
					result = true;
				}
		}
		return result;
	}
	
	public boolean isBracketedExpression(String expression)
	{
		String[] splitExpression = expression.split(" ");
		String[] brackets = splitBrackets(expression);
		
		boolean result = false;
		
		if ((isBracketed(brackets[0].trim())) && (isBracketed(brackets[1].trim())))
		{
			if (isValidBasicExpression(removeBrackets(brackets[0])) && isValidBasicExpression(removeBrackets(brackets[1])))
			{
				String operator = splitExpression[3];
				if (isValidOperator(operator))
				{
					result = true;
				}
			}
		}
		
		return result;
		
	}
	
	public float evaluateExpression(String stringOperand1,String operator, String stringOperand2) 
	{
		float operand1 = Float.parseFloat(stringOperand1);
		float operand2 = Float.parseFloat(stringOperand2);
		return evaluateExpression(operand1, operator, operand2);
	}
	
	public String removeBrackets(String expression)
	
	{
		expression = expression.trim();
		String result = expression.substring(1, expression.length() - 1);
		return result;
	}
	/**
	 * 
	 * @param operand1
	 * @param operator
	 * @param operand2
	 * @return check if the operands are integers, if so then the calculation can take 
	 * place as normal
	 * else the operands are repeatedly multiplied by 10 until they are integers and this 
	 * multiplied factor is returned. If the operator is additive, then both the operands
	 * are multiplied by the largest factor and then the total is divided by that factor
	 * If the operand is a times, both operands are multiplied by their respective 
	 * return factor and then the total is divided by both return factors multiplied
	 * together 
	 * If the operand is divide, both operands are first multiplied by the largest factor
	 * These extra steps are to reduce any rounding errors
	 */
	
	public float evaluateExpression(Float operand1,String operator, Float operand2) 
	{
		float result = 0;
		
		int returnFactor1 = returnFactor(operand1);
		int returnFactor2 = returnFactor(operand2);
		if ((returnFactor1 == 1) && (returnFactor2 == 1))
		{
			switch (operator)
			{
				case "*":
				{
	//				double doubleResult = operand1 * operand2;
					result =  operand1 * operand2;
	//				result = (float)doubleResult;
					break;
				}
				case "-":
				{
//					int returnFactor1 = returnFactor(operand1);
//					int returnFactor2 = returnFactor(operand2);
//					if ((returnFactor1 == 1) && (returnFactor2 == 1))
//					{
						result = operand1 - operand2;
//					}
//					else
//					{
//						int largestFactor = calculateLargestFactor(returnFactor1, returnFactor2);
//						result = ((operand1 * largestFactor) - (operand2 * largestFactor)) / largestFactor;
//					}
					break;
				}
				case "/":
				{
					result = operand1 / operand2;
					break;
				}
				case "+":
				{
					result = operand1 + operand2;
					break;
				}
			}
		}
		else
		{
			int largestFactor = calculateLargestFactor(returnFactor1, returnFactor2);
			switch (operator)
			{
				case "*":
				{
					result =  ((operand1 * returnFactor1) * (operand2 * returnFactor2)) / calculateMultiplicativeFactor(returnFactor1, returnFactor2);
					break;
				}
				
				case "/":
				{
					result =  ((operand1 * largestFactor) / (operand2 * largestFactor));
					break;
				}
				
				case "-":
				{
					result =  ((operand1 * largestFactor) - (operand2 * largestFactor)) / largestFactor;
					break;
				}
				
				case "+":
				{
					result =  ((operand1 * largestFactor) + (operand2 * largestFactor)) / largestFactor;	
					break;
				}
				
			}
		}
		return result;
	}
	
	public float evaluateExpression(Float operand1,String operator, String StringOperand2) 
	{
		float operand2 = Float.parseFloat(StringOperand2);
		
		return evaluateExpression(operand1, operator, operand2);

	}
	
	public String[] splitBrackets(String expression)
	{
		String[] brackets = new String[2];
		brackets[0] = "";
		brackets[1] = "";
		boolean concatNow = true;
		boolean concatAgain = false;
		for (int i = 0; i < expression.length(); i++)
		{
			if (concatNow)
			{
				brackets[0] = brackets[0] + expression.charAt(i);
			}
			if (i != 0)
			{
					
				if ((expression.charAt(i - 1) == ')'))
				{
					concatNow = false;
				}
				else if (expression.charAt(i) == '(')
				{
					concatAgain = true;
				}
				
				if (concatAgain)
				{
					brackets[1] = brackets[1] + expression.charAt(i);
				}
			}
		}
		
		return brackets;
	}
	
	public boolean isLongExpression(String expression)
	{
		boolean result = false;
		String[] splitExpression = expression.split(" ");
		if ((isValidOperand(splitExpression[0]) && checkRepeatedSpace(expression)))
			{
			for (int i = 0; i < splitExpression.length; i++)
			{
				if ((i != splitExpression.length - 1) && ((i == 0) || (result == true)))
				{
					if (isValidOperand(splitExpression[i]))
					{
						if (isValidOperator(splitExpression[i + 1]))
						{
							result = true;
						}
						else 
						{
							result = false;
						}
					}
					else if (isValidOperator(splitExpression[i]))
					{
						if (isValidOperand(splitExpression[i + 1]))
						{
							result = true;
						}
						else
						{
							result = false;
						}
					}
				}
			}
		}
	return result;
	}
	
	public boolean checkRepeatedSpace(String expression)
	{
		boolean result = true;
		for (int i = 0; i < expression.length() - 1; i++)
		{
			if (expression.charAt(i) == ' ')
			{
				if (expression.charAt(i) == expression.charAt(i + 1))
				{
					result = false;
				}
			}
			
		}
		return result;
		
	}
	public float solveLongExpression(String expression)
	{
		float result = 0;
		int iToMiss = 92;
		String[] splitExpression = expression.split(" ");
		int sizeOfAdaptedArray = splitExpression.length;
		for (int i = splitExpression.length - 1; i >= 0; i--)
		{
			if ((i == splitExpression.length - 1) || !(stack.size() == sizeOfAdaptedArray))
				
				if (i != iToMiss)
					{
					if (isValidOperand(splitExpression[i]))
					{
						stack.push(splitExpression[i]);
					}
					else if (isMultiplicative(splitExpression[i]))
					{
						stack.push(splitExpression[i]);
						stack.push(splitExpression[i - 1]);
						stack.push(evaluateExpression(stack.pop(), stack.pop(), stack.pop()));
						sizeOfAdaptedArray -= 2;
						iToMiss = i - 1;
					}
					else if (isAdditive(splitExpression[i]))
					{
						stack.push(splitExpression[i]);
					}
				}
			
			
		}
		result = recursivePop(Float.parseFloat((stack.pop())));
		return result;
	}
	
	public boolean isMultiplicative(String operand)
	{
		if (operand.equals("*"))
		{
			return true;
		}
		else if (operand.equals("/"))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	
	public boolean isAdditive(String operand)
	{
		if (operand.equals("+"))
		{
			return true;
		}
		else if (operand.equals("-"))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	
	public float recursivePop(float result)
	{
		if (stack.isEmpty())
		{
			return result;
		}
		else
		{
			stack.push(result);
			return recursivePop(evaluateExpression(stack.pop(), stack.pop(), stack.pop()));
		}
	}
	
	public int returnFactor(float number)
	{
		int i = 1;
		boolean found = false;
		while ((!found))
		{
			if ((number % 1) > 0)
			{
				number = number * 10;
				i = i * 10;
			}
			else 
			{
				found = true;
			}
		}
		return i;
	}
	
	public int returnFactor(String number)
	{
		return returnFactor(Float.parseFloat(number));
	}
	
	public int calculateLargestFactor(int factor1, int factor2)
	{
		if (factor1 >= factor2)
		{
			return factor1;
		}
		else
		{
			return factor2;
		}
	}
	
	
	
	public int calculateMultiplicativeFactor(int factor1, int factor2)
	{
		return factor1 * factor2;
	}
	
	
	
}
