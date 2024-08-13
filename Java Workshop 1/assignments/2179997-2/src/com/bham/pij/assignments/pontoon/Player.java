package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997
import java.util.ArrayList;

public class Player
{
    private ArrayList<Card> hand = new ArrayList<>();
    //name attribute of player
    private String name;

    //constructor
    public Player(String name)
    {
        this.name = name;
    }

    //getter and setter
    public String getName()
    {
        return name;
    }
    //deals card to player
    public void dealToPlayer(Card card)
    {
        hand.add(card);
    }

    //removes a card from the player's hand
    public void removeCard(Card card)
    {
        hand.remove(card);
    }

    //checks if there is an ace and then adds ten to the value of the deck every time there
    //is one
    public ArrayList<Integer> getNumericalHandValue()
    {
        int numOfAces = 0;
        int handTotal = 0;
        ArrayList<Integer> numericalHandValue = new ArrayList<>();
        for (int i = 0; i < hand.size(); i++)
        {
            int value = hand.get(i).getNumericalValue().get(0);
            handTotal = handTotal + value;
            if (value == 1)
            {
                numOfAces ++;
            }
        }
        numericalHandValue.add(handTotal);

        for (int i = 1; i <= numOfAces; i ++)
        {
            numericalHandValue.add(handTotal + 10*i);
        }
//        if (numOfAces != 0)
//        {
//           numericalHandValue.add(handTotal + 10*numOfAces);
//        }
        return numericalHandValue;
    }
    //returns the largest possible score of the player's hand
    public int getBestNumericalHandValue()
    {
//        if (getNumericalHandValue().size() == 1)
//        {
//            return getNumericalHandValue().get(0);
//        }
//        else
//        {
//            return getNumericalHandValue().get(1);
//        }
        return getNumericalHandValue().get(getNumericalHandValue().size() - 1);
    }
    //returns the hand
    public ArrayList<Card> getCards()
    {
        return this.hand;
    }
    //returns the size of the hand
    public int getHandSize()
    {
        return hand.size();
    }

    public void displayCards()
    {
        System.out.println(getName() + "'s hand: ");
        for(int i = 0; i < getHandSize(); i++)
        {
            System.out.println(hand.get(i).getValue() + " OF " + hand.get(i).getSuit());
        }
        System.out.println();
    }

    public void resetCards()
    {
        getCards().clear();
    }


















}
