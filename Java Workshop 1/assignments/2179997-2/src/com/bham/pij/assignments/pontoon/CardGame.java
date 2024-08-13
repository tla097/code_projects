package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997
import java.util.ArrayList;
public abstract class CardGame
{
    private Deck deck;
    private ArrayList<Player> players = new ArrayList<>();
    private int numberOfPlayers;

    public CardGame(int nPlayers)
    {
        numberOfPlayers = nPlayers;
        for (int i = 0; i < nPlayers; i++)
        {
            players.add(new Player("Player " + (i + 1)));
        }
        deck = new Deck();
    }

    public abstract void dealInitialCards();

    public abstract int compareHands(Player hand1, Player hand2);

    public Deck getDeck()
    {
        return deck;
    }

    public Player getPlayer(int i)
    {
        return players.get(i);
    }

    public int getNumPlayers()
    {
        return numberOfPlayers;
    }

    public void setPlayers(ArrayList<Player> players)
    {
        this.players = players;
    }

    public ArrayList<Player> getPlayers()
    {
        return this.players;
    }

    public void resetCards()
    {
        getDeck().reset();
        for (int i = 0; i < getNumPlayers(); i++)
        {
            getPlayer(i).resetCards();
        }
    }







}
