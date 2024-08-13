package com.bham.fsd.assignments.jabberserver;

import java.io.Serializable;
import java.util.ArrayList;

public class JabberMessage implements Serializable {

	private static final long serialVersionUID = 1L;
	private String message;
	private ArrayList<ArrayList<String>> response = new ArrayList<ArrayList<String>>();
	
	public JabberMessage(String message) {

		this.message = message;
		response = null;
	}
	
	public JabberMessage(String message, ArrayList<ArrayList<String>> data) {

		this.message = message;
		response = data;
	}
	
	public ArrayList<ArrayList<String>> getData() {
		return response;
	}

	public String getMessage() {
		return message;
	}
}
