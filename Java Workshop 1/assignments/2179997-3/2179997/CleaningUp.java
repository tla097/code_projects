package com.bham.pij.assignments.candidates;
//2179997
//Thomas Armstrong

import java.io.*;
import java.nio.Buffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;
import java.util.ArrayList;


public class CleaningUp
{



    private Path dirtyCVPath;
    private String dirtyCVStrPath;
    private Path cleanCVPath;
    private String cleanCVStrPath;

    private ArrayList<String> informationArray = new ArrayList<String>();

    public CleaningUp()
    {
        this.dirtyCVStrPath = new File("dirtycv.txt").getAbsolutePath();
        this.dirtyCVPath = Paths.get(this.dirtyCVStrPath);
        this.cleanCVStrPath = new File("cleancv.txt").getAbsolutePath();
        this.cleanCVPath = Paths.get(this.cleanCVStrPath);
    }

    public String getDirtyCVStrPath()
    {
        return dirtyCVStrPath;
    }

    public Path getDirtyCVPath()
    {
        return dirtyCVPath;
    }

    public Path getCleanCVPath()
    {
        return cleanCVPath;
    }

    public void cleanUpFile()
    {
        ArrayList<String> informationArray = getInformationArray();
        try (BufferedWriter writer =  Files.newBufferedWriter(getCleanCVPath()))
        {
            for (int i = 0; i < informationArray.size(); i++)
            {
                if(informationArray.get(i).equals("Complete"))
                {
                    writer.newLine();
                }
                else
                {
                    writer.write(informationArray.get(i) + ",", 0, informationArray.get(i).length() + 1);
                }
            }
        }
        catch (IOException e)
        {
            System.out.println("We apologise there has been an error");
            System.err.println("Message" + e.getMessage());
        }
    }

    public String getID(String IDNum, String surname)
    {
        String oString = "";
        for (int i = 0; i < 4-IDNum.length(); i++)
        {
            oString = 0 + oString;
        }
        return  surname + oString + IDNum;
    }
    
    public ArrayList<String> getInformationArray()
    {
        ArrayList<String> informationArray = new ArrayList<String>();
        try 
                (Scanner reader = new Scanner(getDirtyCVPath());)
        {
            while (reader.hasNext())
            {
                String[] line;
                String nextLine = reader.nextLine();
                while (!(nextLine.equals("End")))
                {
                    line = nextLine.split(":");
                    String IDNum = null;
                    if ((line[0].contains("CV")))
                        {
                            IDNum = line[0].split(" ")[1];
                            line = reader.nextLine().split(":");
                        }

                    if ((line[0].equals("Surname")))
                    {
                        informationArray.add(getID(IDNum, line[1]));
                        line = reader.nextLine().split(":");
                    }
                    else if((line[0].equals("Qualification")) && (line[1].equals("None")))
                    {
                        informationArray.add("None");
                        line = reader.nextLine().split(":");
                    }

                    if((line[0].equals("Position")) && (line[1].equals("None")))
                    {
                        line = reader.nextLine().split(":");
                    }

                    if((line[0].equals("Experience")) && (line[1].equals("None")))
                    {
                        line = reader.nextLine().split(":");
                    }

                    while ((line[0].replaceAll(" ", "")).equals("FirstName") || (line[0].equals("Address")))
                    {
                        line = reader.nextLine().split(":");
                    }

                    informationArray.add(line[1]);
                    nextLine = reader.nextLine();
                }

                informationArray.add("Complete");


            }

            return informationArray;
        }
        catch(IOException e)
        {
            System.err.println("Error message:  " + e.getMessage());
            System.out.println("We apologise there was an error.");
            return informationArray;
        }
    }


}
