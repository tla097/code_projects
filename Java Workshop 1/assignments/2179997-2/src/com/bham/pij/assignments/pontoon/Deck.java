package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997
import java.util.ArrayList;
import java.util.Random;

public class Deck
{
    //deck of cards
    ArrayList<Card> deck = new ArrayList<>();

    public Deck()
    {
        createDeck();
    }

    public void createDeck()
    {
        for (Card.Suit suit : Card.Suit.values())
        {
            for (Card.Value value : Card.Value.values())
            {
                Card card = new Card(suit, value);
                //card.setSuit(suit);
                //card.setValue(value);
                deck.add(card);
            }
        }
    }

    public void reset()
    {
        deck.clear();
        createDeck();
    }

    public void shuffle()
    {
        Random rand = new Random();
        for (int i = 0; i < deck.size(); i++)
        {
            int randomNumber = rand.nextInt(deck.size() - i) + i;
//            Card tempCard = new Card();
            Card tempCard;
            tempCard = deck.get(i);
            deck.set(i, deck.get(randomNumber));
            deck.set(randomNumber, tempCard);


        }
    }

    public Card getCard(int i)
    {
        return deck.get(i);
    }

    public Card dealRandomCard()
    {
        Random rand = new Random();
        int randomNumber = rand.nextInt(deck.size());
//        Card result = new Card();
        Card result;
        result = deck.get(randomNumber);
        deck.remove(randomNumber);
        return result;
    }

    public int size()
    {
        return deck.size();
    }






}
