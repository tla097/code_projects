package com.bham.pij.assignments.candidates;
//2179997
//Thomas Armstrong
public class JobCandidatesMain
{
    public static void main(String args[])
    {
        CleaningUp clean = new CleaningUp();
        clean.cleanUpFile();

        CandidatesToInterview interview = new CandidatesToInterview(clean.getCleanCVPath());
        interview.findCandidates();
        interview.candidatesWithExperience();
        interview.createCSVFile();
        interview.createReport();
    }
}
