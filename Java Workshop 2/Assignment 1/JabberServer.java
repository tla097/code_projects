package com.bham.fsd.assignments.jabberserver;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.PreparedStatement;

public class JabberServer {
	
	private static String dbcommand = "jdbc:postgresql://127.0.0.1:5432/postgres";
	private static String db = "postgres";
	private static String pw = "";

	private static Connection conn;
	
	public static Connection getConnection() {
		return conn;
	}

	public static void main(String[] args) {
				
		JabberServer jabber = new JabberServer();
		JabberServer.connectToDatabase();
		jabber.resetDatabase(); 	
		
		/*
		 * Put calls to your methods here to test them.
		 */
	}
	
	public ArrayList<String> getFollowerUserIDs(int userid) {
		

		/*
		 * Add your code to this method here.
		 * Remove the 'return null' statement and add your own return statement.
		 */
		
		return null;
	}

	public ArrayList<String> getFollowingUserIDs(int userid) {

		/*
		 * Add your code to this method here.
		 * Remove the 'return null' statement and add your own return statement.
		 */
			
		return null;
	}
	
	public ArrayList<ArrayList<String>> getMutualFollowUserIDs() {

		/*
		 * Add your code to this method here.
		 * Remove the 'return null' statement and add your own return statement.
		 */
		
		return null;
	}

	public ArrayList<ArrayList<String>> getLikesOfUser(int userid) {
		
		/*
		 * Add your code to this method here.
		 * Remove the 'return null' statement and add your own return statement.
		 */
		
		return null;
	}
	
	public ArrayList<ArrayList<String>> getTimelineOfUser(int userid) {
		
		/*
		 * Add your code to this method here.
		 * Remove the 'return null' statement and add your own return statement.
		 */
		
		return null;
	}

	public void addJab(String username, String jabtext) {
		
		/*
		 * Add your code to this method here.
		 */
	}
	
	public void addUser(String username, String emailadd) {
		
		/*
		 * Add your code to this method here.
		 */
	}
	
	public void addFollower(int userida, int useridb) {
		
		/*
		 * Add your code to this method here.
		 */
	}
	
	public void addLike(int userid, int jabid) {
		/*
		 * Add your code to this method here.
		 */
	}
	
	public ArrayList<String> getUsersWithMostFollowers() {
		
		/*
		 * Add your code to this method here.
		 * Remove the 'return null' statement and add your own return statement.
		 */
		
		return null;
	}
	
	public JabberServer() {}
	
	public static void connectToDatabase() {

		try {
			conn = DriverManager.getConnection(dbcommand,db,pw);

		}catch(Exception e) {		
			e.printStackTrace();
		}
	}

	/*
	 * Utility method to print an ArrayList of ArrayList<String>s to the console.
	 */
	private static void print2(ArrayList<ArrayList<String>> list) {
		
		for (ArrayList<String> s: list) {
			print1(s);
			System.out.println();
		}
	}
		
	/*
	 * Utility method to print an ArrayList to the console.
	 */
	private static void print1(ArrayList<String> list) {
		
		for (String s: list) {
			System.out.print(s + " ");
		}
	}

	public void resetDatabase() {
		
		dropTables();
		
		ArrayList<String> defs = loadSQL("jabberdef");
	
		ArrayList<String> data =  loadSQL("jabberdata");
		
		executeSQLUpdates(defs);
		executeSQLUpdates(data);
	}
	
	private void executeSQLUpdates(ArrayList<String> commands) {
	
		for (String query: commands) {
			
			try (PreparedStatement stmt = conn.prepareStatement(query)) {
				stmt.executeUpdate();
			} catch (SQLException e) {
				e.printStackTrace();
			}
		}
	}

	private ArrayList<String> loadSQL(String sqlfile) {
		
		ArrayList<String> commands = new ArrayList<String>();
		
		try {
			BufferedReader reader = new BufferedReader(new FileReader(sqlfile + ".sql"));
			
			String command = "";
			
			String line = "";
			
			while ((line = reader.readLine())!= null) {
				
				if (line.contains(";")) {
					command += line;
					command = command.trim();
					commands.add(command);
					command = "";
				}
				
				else {
					line = line.trim();
					command += line + " ";
				}
			}
			
			reader.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return commands;
		
	}

	private void dropTables() {
		
		String[] commands = {
				"drop table jabberuser cascade;",
				"drop table jab cascade;",
				"drop table follows cascade;",
				"drop table likes cascade;"};
		
		for (String query: commands) {
			
			try (PreparedStatement stmt = conn.prepareStatement(query)) {
				stmt.executeUpdate();
			} catch (SQLException e) {
				e.printStackTrace();
			}
		}
	}
}
