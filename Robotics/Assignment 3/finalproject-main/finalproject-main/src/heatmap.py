#!/usr/bin/env python3
import map
from random import randrange
import random
import copy

class heatmap:
    # will not have many people in the store so I'll scrap the vector frequency
    def __init__(self, nrPeople, box_locations):
        # initial position of people - I think about 40-60 people is good
        self.people = []
        self.box_locations = box_locations
        self.MAX_PER_STATE = 3
        for _ in range(nrPeople):
            self.people.append(self.newPerson())

        self.possible_transitions = map.possible_transitions()

        # print("people ", self.people)

    def newPerson(self):
        while True:
            wantedState = randrange(len(map.states()) - 1)
            while(wantedState in self.box_locations):
                wantedState = randrange(len(map.states()) - 1)
            if self.people.count(wantedState) >= self.MAX_PER_STATE:
                continue
            return wantedState

    def move(self, iterations):
        possible_transitions = self.possible_transitions # I think it generally removes somehow idk python so saving a copy
        people = self.people
        # print(people)
        for _ in range(iterations):
            copy_possible_transitions = copy.deepcopy(possible_transitions)
            iterObj = iter(range(len(people)))
            i = next(iterObj)
            while True: # this is a normal iteration from 0 to nr of people 
                try:
                    state = "s" + str(people[i])
                    currentPerson = people[i]
                    if(copy_possible_transitions[state] != ['TERMINAL']):

                        if(copy_possible_transitions[state] == []): # this is so it doens't crash because of random
                            i = next(iterObj)
                            continue    # if all the transitions are exhausted stay where you are                      
                        
                        rn = random.choice(copy_possible_transitions[state])
                        # print(f"random choice for {people[i]}: {rn}")


                        if (rn == 'LEFT'):
                            if(people.count(people[i] - 1) >= self.MAX_PER_STATE): # this means if the state in the left is >= 3
                                # print(f"State {people[i] - 1} is full, this is the full people table {sorted(self.people)}")
                                copy_possible_transitions[state].remove(rn)
                                continue
                            people[i] -=1
                            
                        if(rn == 'RIGHT'):
                            if(people.count(people[i] + 1) >= self.MAX_PER_STATE):
                                # print(f"State {people[i] + 1} is full, this is the full people table {sorted(self.people)}")
                                copy_possible_transitions[state].remove(rn)
                                continue
                            people[i] +=1

                        if(rn == 'DOWN'):
                            people[i] = self.moveDown("s" + str(people[i]))
                            if(currentPerson == people[i]):
                                copy_possible_transitions[state].remove(rn)
                                continue

                        if(rn == 'UP'):
                            people[i] = self.moveUp("s" + str(people[i]))
                            if(currentPerson == people[i]):
                                copy_possible_transitions[state].remove(rn)
                                continue
      
                    else:
                        people[i] = self.newPerson() #assume they finish the shopping when they get to term state and summon new ones
                    i = next(iterObj) # go to the next person

                except StopIteration:
                    break
            # print(copy_possible_transitions)
        return people

    #this is the function that you should call when you get to a goal state and you need to update the heatmap and recalculate the policy
    #iterations = number of steps the robot moves (would make sense)
    def stateUncertainty (self, iterations):
        self.people = self.move(iterations)
        return self.people


    def moveUp(self,state):
        allStates = map.states()
        x = allStates[state][0]
        y = allStates[state][1] + 1
        newState = state
        for nextState in allStates:
            if(allStates[nextState][0] == x and allStates[nextState][1] == y):
                newState = int(''.join(filter(str.isdigit, nextState)))
                break

        if (self.people.count(newState) >= self.MAX_PER_STATE):
            # print(f"State {newState} is full, this is the full people table {sorted(self.people)}")
            return int(''.join(filter(str.isdigit, state)))
        return int(''.join(filter(str.isdigit, nextState)))

    def moveDown(self,state): # return the new state if it doens't already have more then self.MAX_PER_STATE
        allStates = map.states()
        x = allStates[state][0]
        y = allStates[state][1] - 1
        newState = state
        for nextState in allStates:
            if(allStates[nextState][0] == x and allStates[nextState][1] == y):
                newState = int(''.join(filter(str.isdigit, nextState)))
                break
        if (self.people.count(newState) >= self.MAX_PER_STATE):
            # print(f"State {newState} is full, this is the full people table {sorted(self.people)}")
            return int(''.join(filter(str.isdigit, state)))

        return int(''.join(filter(str.isdigit, nextState)))

