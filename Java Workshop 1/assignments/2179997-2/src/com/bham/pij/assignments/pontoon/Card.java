package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997

import java.util.ArrayList;

/**
 * @author tom
 * @version 1.0
 */

public class Card {
    // suit enum
    public static enum Suit{HEARTS, SPADES, CLUBS, DIAMONDS};
    //value enum
    public static enum Value{ACE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING};
    /**
     * @variables for the suit and value of the card
     */
    private Suit suit;
    private Value value;

    public Card(Suit suit, Value value)
    {
        this.suit = suit;
        this.value = value;
    }

    //getters and setters
    public Suit getSuit()
    {
     return this.suit;
    }

    public void setSuit(Suit suit)
    {
        this.suit = suit;
    }

    public Value getValue()
    {
        return this.value;
    }

    public void setValue(Value value)
    {
        this.value = value;
    }

    public ArrayList<Integer> getNumericalValue()
    {
        ArrayList<Integer> numericalValue = new ArrayList<Integer>();
        switch (getValue())
        {
            case ACE:
            {
                numericalValue.add(1);
                numericalValue.add(11);
                break;
            }
            case TWO:
            {
                numericalValue.add(2);
                break;
            }
            case THREE:
            {
                numericalValue.add(3);
                break;
            }
            case FOUR:
            {
                numericalValue.add(4);
                break;
            }
            case FIVE:
            {
                numericalValue.add(5);
                break;
            }
            case SIX:
            {
                numericalValue.add(6);
                break;
            }
            case SEVEN:
            {
                numericalValue.add(7);
                break;
            }
            case EIGHT:
            {
                numericalValue.add(8);
                break;
            }
            case NINE:
            {
                numericalValue.add(9);
                break;
            }
            default :
            {
                numericalValue.add(10);
            }
        }
        return numericalValue;
    }

}
