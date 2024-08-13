package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997
import java.util.ArrayList;

public class Stack
{
    ArrayList<Player> stackArray = new ArrayList<Player>();

    public Player pop()
    {
        Player player = stackArray.get(0);
        stackArray.remove(0);
        return player;
    }

    public void push(Player player)
    {
        stackArray.add(0, player);
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