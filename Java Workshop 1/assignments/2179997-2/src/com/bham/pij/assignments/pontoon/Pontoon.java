package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997
import java.util.ArrayList;

public class Pontoon extends CardGame
{
//    public Pontoon(ArrayList<Player> players)
//    {
//        super(players.size());
//        setPlayers(players);
//    }

    public Pontoon(int nPlayers)
    {
        super(nPlayers);
    }



    @Override
    public void dealInitialCards()
    {
        for (int playerIndex = 0; playerIndex < getNumPlayers(); playerIndex++)
        {
            for (int cardIndex = 0; cardIndex < 2; cardIndex++)
            {
                getPlayers().get(playerIndex).dealToPlayer(getDeck().dealRandomCard());
            }
        }
    }

    @Override
    public int compareHands(Player hand1, Player hand2)
    {
        int outcome1, outcome2;
        outcome1 = determineOutCome(hand1);
        outcome2 = determineOutCome(hand2);
        if (outcome1 < outcome2)
        {
            return -1;
        }
        else if (outcome2 < outcome1)
        {
            return 1;
        }
        else if ((outcome1 == 4) || (outcome1 == 2))
        {
            return getLargerHand(hand1, hand2);
        }
        else
            return 0;
    }

    public boolean isBust(Player player)
    {
        if (player.getNumericalHandValue().get(0) > 21)
        {
            return true;
        }
        else return false;
    }

    public boolean isPontoon(Player player)
    {
        if (player.getHandSize() == 2)
        {
            if (player.getBestNumericalHandValue() == 21)
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }

    public boolean isFiveCardTrick(Player player)
    {
        boolean result = false;

        if (player.getHandSize() == 5)
        {
//            if ((player.getBestNumericalHandValue() < 22) || (player.getNumericalHandValue().get(0) < 22))
//            {
//                result = true;
//            }
            if (getLargerScore(player) < 22)
            {
                result = true;
            }
        }
        return result;
    }

    public boolean is21Total(Player player)
    {
        if (getLargerScore(player) == 21)
        {
            return true;
        }
        else
            return false;
    }

    public int determineOutCome(Player player)
    {
        if (!isBust(player))
        {
            if (isPontoon(player))
                return 1;
            else if (isFiveCardTrick(player))
                return 2;
            else if (is21Total(player))
                return 3;
            else
                return 4;
        }
        else return 5;
    }

    public int getLargerScore(Player player)
    {
//        if ((player.getBestNumericalHandValue() > player.getNumericalHandValue().get(0)) && (player.getBestNumericalHandValue() <22))
//        {
//            return player.getBestNumericalHandValue();
//        }
//        else
//        {
//            return player.getNumericalHandValue().get(0);
//        }
        boolean found = false;
        int result = 0;
        for(int i = player.getNumericalHandValue().size() - 1; i >= 0; i --)
        {
            if ((player.getNumericalHandValue().get(i) < 22) && !found) {
                found = true;
                result = player.getNumericalHandValue().get(i);
            }
        }
        return result;
    }

    public int getLargerHand(Player player1, Player player2)
    {
        int player1Score, player2Score;
        player1Score = getLargerScore(player1);
        player2Score = getLargerScore(player2);
        if (player1Score > player2Score)
            return -1;
        else if (player2Score > player1Score)
            return 1;
        else
            return 0;
    }

    public boolean mustTwist(Player player)
    {
        boolean result = false;

        if (player.getBestNumericalHandValue() < 16)
        {
            result = true;
        }
        else
        {
            for (int i = player.getNumericalHandValue().size() - 1; i > 0; i--)
            {
                if (player.getNumericalHandValue().get(i) > 21 && (player.getNumericalHandValue().get(i - 1)) < 16)
                {
                    result = true;
                }
            }
        }
//        if (player.getBestNumericalHandValue() == player.getNumericalHandValue().get(0))
//        {
//            if (player.getBestNumericalHandValue() < 16)
//            {
//                result = true;
//            }
//        }
//        else if ((player.getNumericalHandValue().get(0) < 16) && (player.getBestNumericalHandValue() > 21))
//        {
//            result = true;
//        }

        return result;
    }

    public void twist(Player player)
    {
        player.dealToPlayer(getDeck().dealRandomCard());
    }


//    public boolean canStick(Player player)
//    {
//        if ((player.getBestNumericalHandValue() > 15 && player.getBestNumericalHandValue() < 22)
//            ||(player.getNumericalHandValue().get(0) > 15 && player.getNumericalHandValue().get(0) < 22))
//        {
//            return true;
//        }
//        else
//            return false;
//    }













}
