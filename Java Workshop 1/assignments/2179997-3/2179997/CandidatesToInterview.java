package com.bham.pij.assignments.candidates;
//2179997
//Thomas Armstrong

import java.awt.image.AreaAveragingScaleFilter;
import java.io.*;
import java.io.IOException;
import java.lang.annotation.RetentionPolicy;
import java.nio.Buffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class CandidatesToInterview
{

    private final String [] keywordsDegree = {"Degree in Computer Science", "Masters in Computer Science"};
    private final String [] keywordsExperience = {"Data Analyst", "Programmer", "Computer programmer", "Operator"};
    private final String[] headings = {"Identifier", "Qualification", "Position", "Experience", "eMail"};
    private Path toInterviewPath;
    private String toInterviewStrPath;
    private Path cleanCVPath;
    ArrayList<String> cleanCVInfo = new ArrayList<String>();
    ArrayList<ArrayList<String>> experienceInterviewList = new ArrayList<ArrayList<String>>();




    public Path getToInterviewPath()
    {
        return toInterviewPath;
    }

    public String getToInterviewStrPath()
    {
        return toInterviewStrPath;
    }







    public ArrayList<String> getCleanCVInfo()
    {
        return cleanCVInfo;
    }

    public void setCleanCVInfo()
    {
        try (Scanner reader = new Scanner(getCleanCVPath()))
        {
            while (reader.hasNext())
            {
                this.cleanCVInfo.add(reader.nextLine());
            }
        }
        catch (IOException e)
        {
            System.out.println("We apologise there has been an error");
            System.err.println("Message" + e.getMessage());
        }
    }






    public CandidatesToInterview(Path cleanCVPath)
    {
        this.toInterviewStrPath = new File("to-Interview.txt").getAbsolutePath();
        this.toInterviewPath = Paths.get(toInterviewStrPath);

        this.cleanCVPath = cleanCVPath;
    }

    public void findCandidates()
    {
        setCleanCVInfo();
        filterList();
        try (BufferedWriter writer = Files.newBufferedWriter(getToInterviewPath()))
        {
            for (int i = 0; i < getCleanCVInfo().size(); i++)
            {
                writer.write(getCleanCVInfo().get(i), 0, getCleanCVInfo().get(i).length());
                writer.newLine();
            }
        }
        catch (IOException e)
        {
            System.out.println("We apologise there has been an error");
            System.err.println("Message" + e.getMessage());
        }
    }

    public void filterList()
    {
        for (int i = 0; i < getCleanCVInfo().size(); i++)
        {
            if (!((isCandidate(getCleanCVInfo().get(i), getKeywordsDegree())) && ((isCandidate(cleanCVInfo.get(i), getKeywordsExperience())))))
            {
                getCleanCVInfo().remove(i);
                i --;
            }
            else
            {
                getCleanCVInfo().set(i, commasToSpaces(getCleanCVInfo().get(i)));
            }
        }
    }

    public String commasToSpaces(String CVLine)
    {
        return CVLine.replaceAll(",", " ");
    }



    public String[] getKeywordsDegree()
    {
        return keywordsDegree;
    }

    public String[] getKeywordsExperience()
    {
        return keywordsExperience;
    }

    public Path getCleanCVPath()
    {
        return cleanCVPath;
    }
    public boolean isCandidate(String candidateCV, String[] keywordArray)
    {
        boolean result = false;
        for (int i = 0; i < keywordArray.length; i++)
        {
            if (candidateCV.contains(keywordArray[i]))
            {
                result = true;
            }
        }
        return result;


    }

    public void candidatesWithExperience()
    {

        String toIntervieEpxerienceStrPath = new File("to-Interview-Experience.txt").getAbsolutePath();
        Path toInterviewExperiencePath = Paths.get(toIntervieEpxerienceStrPath);
        intitialiseExperienceInterviewList(getInterviewList1D());
        ArrayList<ArrayList<String>> experienceArray = getExperienceArray();
        try (BufferedWriter writer = Files.newBufferedWriter(toInterviewExperiencePath))
        {
            for (int i = 0; i < experienceArray.size(); i++)
            {
                String interviewLine = "";
                for (int j = 0; j < experienceArray.get(i).size(); j++)
                {
                    interviewLine = interviewLine + " " + experienceArray.get(i).get(j);
                }
                 writer.write(interviewLine.trim(), 0, interviewLine.trim().length());
                writer.newLine();
            }
        }
        catch (IOException e)
        {
            System.out.println("We apologise there has been an error");
            System.err.println("Message" + e.getMessage());
        }
    }

    public ArrayList<String> getInterviewList1D()
    {
        ArrayList<String> interviewList1D = new ArrayList<String>();
        try (Scanner reader = new Scanner(getToInterviewPath()))
        {
            while(reader.hasNext())
            {
               interviewList1D.add(reader.nextLine());
            }
        }
        catch (IOException e)
        {
            System.out.println("We apologise there has been an error");
            System.err.println("Message" + e.getMessage());
        }
        finally
        {
            return interviewList1D;
        }
    }

    public void intitialiseExperienceInterviewList(ArrayList<String> interviewList1D)
    {
        for (int i = 0; i < interviewList1D.size(); i++)
        {
            ArrayList<String> dummyList = new ArrayList<String>();
            getExperienceInterviewList().add(i, dummyList);
            String[] splitInterveiwList = explicitSplit(interviewList1D.get(i).split(" "));
            for (int j = 0; j < splitInterveiwList.length; j++)
            {
                this.getExperienceInterviewList().get(i).add(splitInterveiwList[j].trim());
            }
        }
    }

    public String[] explicitSplit(String[] splitList)
    {
        int i = 0;
        do
        {
            i ++;
        }
        while(!((splitList[i].equals("Masters")) || (splitList[i].equals( "Degree"))));


        int j = i;

            do
            {
                splitList[j] = splitList[j].trim() + " " + splitList[++i].trim();
            }
            while (!isAcceptable(keywordsDegree, splitList[j]));

        while(!(splitList[i+1].contains("@")))
        {
            j++;
            splitList[j] = "";
            do
            {
                splitList[j] = splitList[j].trim() + " " + splitList[++i];
            }
            while (!isAcceptable(keywordsExperience, splitList[j].trim()));

            splitList[++j] = splitList[++i];
        }

        splitList[++j] = splitList[++i];
        String[] explicitSplit = new String[++j];
        for (int k = 0; k < j; k++)
        {
            explicitSplit[k] = splitList[k];
        }


        return explicitSplit;

    }

    public boolean isAcceptable(String[] array, String testString)
    {
        for (String arrayString : array)
        {
            if (arrayString.equals(testString))
            {
                return true;
            }
        }
        return false;
    }

    public ArrayList<ArrayList<String>> getExperienceInterviewList()
    {
        return this.experienceInterviewList;
    }

    public void setExperienceInterviewList(ArrayList<ArrayList<String>> experienceInterviewList)
    {
        this.experienceInterviewList = experienceInterviewList;
    }

    public ArrayList<ArrayList<String>> getExperienceArray()
    {
        ArrayList<ArrayList<String>> successfulArray = new ArrayList<ArrayList<String>>();
        int k = 0;
        for (int i = 0; i < getExperienceInterviewList().size(); i++)
        {
            for (int j = 0; j < getExperienceInterviewList().get(i).size(); j++)
            {
                String employeeDataElement = getExperienceInterviewList().get(i).get(j);
                if (isNumeric(employeeDataElement))
                {
                    double yearsOfExperience = Double.parseDouble(employeeDataElement);
                    if (yearsOfExperience > 5)
                    {
                        ArrayList<String> singleData = new ArrayList<String>();
                        singleData.add(getExperienceInterviewList().get(i).get(0));
                        singleData.add(employeeDataElement);
                        successfulArray.add(singleData);
                        k++;
                    }
                }
            }
        }
        return successfulArray;
    }

    public boolean isNumeric(String potentialNumber)
    {
        try
        {
            Double.parseDouble(potentialNumber);
            return true;
        }
        catch(NumberFormatException e)
        {
            return false;
        }
    }

    public void createCSVFile()
    {
        ArrayList<String> CSVElements = addFileData(addHeadings());
        try (PrintWriter writer = new PrintWriter("to-interview-table-format.csv"))
        {
            for (int i = 0; i < CSVElements.size(); i++)
            {
                writer.write(CSVElements.get(i));
            }
        }
        catch (FileNotFoundException e)
        {
            System.out.println("We apologise, there has been an error.");
            e.printStackTrace();
        }
    }

    private ArrayList<String> addFileData(ArrayList<String> CSVElements)
    {
        int emailIndex = getEmailIndex(CSVElements);
        for (int i = 0; i < getExperienceInterviewList().size(); i++)
        {
            int currentLastListIndex = getExperienceInterviewList().get(i).size() - 1;
            for (int j = 0; j <= emailIndex; j++)
            {
                if (j == emailIndex)
                {
                    CSVElements.add(getExperienceInterviewList().get(i).get(currentLastListIndex) + "\n");
                }
                else if (j < currentLastListIndex)
                {
                    CSVElements.add(getExperienceInterviewList().get(i).get(j) + ",");
                }
                else
                {
                    CSVElements.add(" ,");
                }
            }
            int lastIndex = CSVElements.size() - 1;
//            CSVElements.set(lastIndex, replaceLast(CSVElements.get(lastIndex)));
        }
        return CSVElements;
    }

    public int getEmailIndex(ArrayList<String> headingsArray)
    {
        for (int i = 0; i < headingsArray.size(); i++)
        {
            if (headingsArray.get(i).contains("eMail"))
            {
                return i;
            }
        }
        return 0;
    }

    public String removeLastNumber(String string)
    {
//        char[] charArray = string.toCharArray();
//        charArray[charArray.length - 1] = '\n';
//        String result = String.valueOf(charArray);
        for (int i = string.length(); i < 0; i--)
        {
            try
            {
                Double.parseDouble(String.valueOf(string.charAt(i)));
            }
            catch(NumberFormatException e)
            {
                return string.substring(0, i);
            }
            return null;
        }
        return string.substring(0, string.length() - 1);



    }

    public ArrayList<String> addHeadings()
    {
        ArrayList<String> CSVElements = new ArrayList<String>();
        for (int i = 0; i < 2; i++)
        {
            CSVElements.add(headings[i] + ",");
        }
        for (int j = 2; j < getNumberOfExperiences() + 2; j++)
        {
            for (int i = 2; i < 4; i++)
            {
                CSVElements.add(headings[i] + Integer.toString(j - 1) + ",");
            }
        }

        CSVElements.add(headings[headings.length - 1] + "\n");

        return CSVElements;

    }

    public int getLongestCV()
    {
        int result = 0;
        for (int i = 0; i < experienceInterviewList.size(); i++)
        {
            if (experienceInterviewList.get(i).size() >= result)
            {
                result = experienceInterviewList.get(i).size();
            }
        }
        return result;
    }

    public int getNumberOfExperiences()
    {
        int result = (getLongestCV()-3)/2;
        return result;
    }

    public void createReport()
    {
        ArrayList<ArrayList<String>> reportArray = makeReportArray();
        for (int column = 0; column < reportArray.get(0).size(); column++)
        {
            int longestElement = getLongestElement(reportArray, column);
            for (int recordNum = 0; recordNum < reportArray.size(); recordNum++)
            {
                reportArray.set(recordNum, formatLength(longestElement, reportArray.get(recordNum),column ));
            }
        }

        for (int recordNum = 0; recordNum < reportArray.size(); recordNum++)
        {
            System.out.printf("%-10s %-10s %-10s %-10s %-10s\n",
                    reportArray.get(recordNum).get(0),
                    reportArray.get(recordNum).get(1),
                    reportArray.get(recordNum).get(2),
                    reportArray.get(recordNum).get(3),
                    reportArray.get(recordNum).get(4));
        }
    }


    public ArrayList<ArrayList<String>> makeReportArray()
    {
        ArrayList<ArrayList<String>> reportArray = new ArrayList<ArrayList<String>>();
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(new File("to-interview-table-format.csv").getAbsolutePath())))
        {
            String nextLine = reader.readLine();
            int j = 0;
            while(nextLine != null)
            {
                String[] splitLine = nextLine.split(",");
                ArrayList<String> lineArray = new ArrayList<>(Arrays.asList(splitLine));
                if (j == 0)
                {
                    for (int i = 2; i < 4; i++)
                    {
                        lineArray.set(i, removeLastNumber(lineArray.get(i)));
                    }
                }
                reportArray.add(lineArray);
                reportArray.get(j).set(4, splitLine[splitLine.length - 1] + "\n");
                for (int i = reportArray.get(j).size() - 1; i > 4; i--)
                {
                    reportArray.get(j).remove(i);
                }
                nextLine = reader.readLine();
                j++;
            }
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        finally
        {
            return reportArray;
        }
    }

    public int getLongestElement(ArrayList<ArrayList<String>> inputArray, int column)
    {
        int length = 0;
        for (int i = 0; i < inputArray.size(); i++)
        {
            if (inputArray.get(i).get(column).length() > length)
            {
                length = inputArray.get(i).get(column).length();
            }
        }
        return length;
    }

    public ArrayList<String> formatLength(int length, ArrayList<String> inputArray, int column)
    {
//        if (column == 2)
//        {
//            while (inputArray.get(column).length() < length + 10)
//            {
//                inputArray.set(column, inputArray.get(column) + " ");
//            }
//        }
//        else
//        {
            while (inputArray.get(column).length() < length + 2)
            {
                inputArray.set(column, inputArray.get(column) + " ");
            }
//        }
//        for (int i = 0; i < 2; i++)
//        {
//            inputArray.set(column, inputArray.get(column) + "\t");
//        }

        return inputArray;
    }


 }
