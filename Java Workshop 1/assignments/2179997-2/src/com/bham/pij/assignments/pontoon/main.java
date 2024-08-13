package com.bham.pij.assignments.pontoon;
//Thomas Armstrong 2179997
 import java.util.ArrayList;
 import java.util.Scanner;
public class main {

    public static Stack playerStack = new Stack();
    public static void main(String args[])
    {
        Scanner sc = new Scanner(System.in);
        String anotherGo;
        int numOfPlayers;
//        ArrayList<Player> players = new ArrayList<>();
//
//        players.clear();
//        addPlayer(players);
//            do {
//                addPlayer(players);
//                System.out.println("Do you want to add another player? (y/n): ");
//                anotherPlayer = sc.nextLine().toLowerCase();
//
//            } while (anotherPlayer.equals("y"));
//
//            Pontoon pontoon = new Pontoon(players);

        System.out.println("How many players do you want to play?: ");
        numOfPlayers = sc.nextInt();
        Pontoon pontoon = new Pontoon(numOfPlayers);
        do
        {


            run(pontoon);
            System.out.println("Do you want to play again?(y/n): ");
            anotherGo = sc.nextLine().trim().toLowerCase();
            pontoon.resetCards();

        }while (anotherGo.equals("y"));


    }

    public static void addPlayer(ArrayList<Player> players)
    {
        Scanner sc = new Scanner(System.in);
        System.out.println("Please enter the name of the player: ");
        Player player = new Player(sc.nextLine().trim());
        players.add(player);
    }

    public static void run(Pontoon pontoon)
    {
        pontoon.getDeck().shuffle();
        System.out.println("Dealing 2 cards for each player");
        pontoon.dealInitialCards();

        pontoon.getPlayer(0).getCards().get(0).setValue(Card.Value.FIVE);
        pontoon.getPlayer(0).getCards().get(1).setValue(Card.Value.FIVE);
        Card card = pontoon.getDeck().dealRandomCard();
        card.setValue(Card.Value.ACE);
        pontoon.getPlayer(0).dealToPlayer(card);
//        Card card2 = pontoon.getDeck().dealRandomCard();
//        card2.setValue(Card.Value.ACE);
//        pontoon.getPlayer(0).dealToPlayer(card2);
//        Card card3 = pontoon.getDeck().dealRandomCard();
//        card3.setValue(Card.Value.ACE);
//        pontoon.getPlayer(0).dealToPlayer(card3);
//
//
//
//        pontoon.getPlayer(1).getCards().get(0).setValue(Card.Value.KING);
//        pontoon.getPlayer(1).getCards().get(1).setValue(Card.Value.THREE);
//
//        Card card = pontoon.getDeck().dealRandomCard();
//        card.setValue(Card.Value.FIVE);
//        Card card2 = pontoon.getDeck().dealRandomCard();
//        card2.setValue(Card.Value.FIVE);
//        Card card3 = pontoon.getDeck().dealRandomCard();
//        card3.setValue(Card.Value.FIVE);
//
//        pontoon.getPlayer(0).dealToPlayer(card);
//        pontoon.getPlayer(0).dealToPlayer(card2);
//        pontoon.getPlayer(0).dealToPlayer(card3);

        startTurns(pontoon);
        loadStack(pontoon);
        System.out.println(winnerIs(pontoon, false));
    }

    public static void loadStack(Pontoon pontoon)
    {
        for (int i = 0; i < pontoon.getNumPlayers(); i++)
        {
            playerStack.push(pontoon.getPlayer(i));
        }
    }


    public static void startTurns(Pontoon pontoon)
    {
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < pontoon.getNumPlayers(); i ++)
        {
            System.out.println(pontoon.getPlayer(i).getName() + "'s turn: ");
            boolean finishGo = false;
            while (!finishGo) {
                pontoon.getPlayer(i).displayCards();
                if (!pontoon.isBust(pontoon.getPlayer(i)))
                {
                    if (pontoon.mustTwist(pontoon.getPlayer(i)))
                    {
                        System.out.println(pontoon.getPlayer(i).getName() + " must twist");
                        pontoon.twist(pontoon.getPlayer(i));
                    }
                    else
                    {
                        boolean validInput = false;
                        while (!validInput)
                        {
                            System.out.println(pontoon.getPlayer(i).getName() + ", do you want to Stick or Twist? (s/t): ");
                            String input = sc.nextLine().trim().toLowerCase();
                            if (input.equals("s"))
                            {
                                finishGo = true;
                                validInput = true;
                            }
                            else if (input.equals("t"))
                            {
                                pontoon.twist(pontoon.getPlayer(i));
                                validInput = true;
                            }
                            else
                            {
                                System.out.println("That was not a valid input");
                            }
                        }
                    }
                }
                else
                {
                    finishGo = true;
                    System.out.println(pontoon.getPlayer(i).getName() + " is bust\n");
                }
            }


        }
    }

    public static String winnerIs(Pontoon pontoon, boolean draw)
    {

        if (playerStack.size() == 1)
        {
            if (draw)
            {
                return "The result is a draw";
            }
            else
            {
                return "The winner is: " + playerStack.pop().getName();
            }
        }
        else
        {
            Player previousVictor = playerStack.pop();
            Player challenger = playerStack.pop();
            int result = pontoon.compareHands(previousVictor, challenger);
            if (result == -1)
            {
                playerStack.push(previousVictor);
                return winnerIs(pontoon, false);
            }
            else if (result == 1)
            {
                playerStack.push(challenger);
                return winnerIs(pontoon, false);
            }
            else
            {
                playerStack.push(previousVictor);
                return winnerIs(pontoon, true);
            }


        }
    }






}
