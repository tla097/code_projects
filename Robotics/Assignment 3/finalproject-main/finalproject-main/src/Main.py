#!/usr/bin/env python3

import rospy
from Box import Box
from Robot import Robot
from Customer import Customer
from Shop import Shop
import map, transitions

def main():
    # initialise boxes with ingredients and quantities
    # note: robot also has a list of oos items passed in (for testing) though they may not match quantities below 
    boxes = []
    boxes.append(Box("Dairy", {"milk": 3, "oat milk": 0, "almond milk": 1}))
    boxes.append(Box("Meat", {"bacon": 1, "pork mince": 1, "ham": 2}))
    boxes.append(Box("Rice", {"jasmine rice": 1, "basmati rice": 1, "long grain rice": 1}))
    boxes.append(Box("Oil", {"sunflower oil": 2, "olive oil": 3, "vegetable oil": 1, "avocado oil": 0}))
    boxes.append(Box("Baking", {"flour": 4, "eggs": 2, "yeast": 1}))
    # boxes.append(Box("Nuts", {"almonds": 0, "cashews": 0, "peanuts": 4})) # replaced by bread box
    boxes.append(Box("Bread", {"white bread": 3, "seeded bread": 2, "bagels": 0}))
    boxes.append(Box("Pasta", {"spaghetti": 3, "penne": 2, "lasagne": 1}))
    boxes.append(Box("Fish", {"salmon": 0, "sea bass": 2}))  # TODO: change salmon back to 1
    boxes.append(Box("Vegetables", {"carrots": 2, "cucumber": 1, "potato": 9}))
    boxes.append(Box("Fruit", {"banana": 4, "apple": 5, "strawberry": 2}))
    boxes.append(Box("Spices", {"paprika": 1, "cumin": 3}))
    boxes.append(Box("Alcohol", {"vodka": 2, "rum": 3}))
    shop = Shop(boxes)

    # goal_ingredients = ["bacon", "salmon", "carrots", "paprika", "oat milk", "bacon", "basmati rice", "eggs", "bagels", "bacon", "avocado oil"]
    goal_ingredients = ["milk", "bacon", "basmati rice", "olive oil", "yeast", "bagels", "penne", "sea bass", "carrots", "banana", "paprika", "vodka"]
    requested_ingredients = goal_ingredients.copy()
    replacements = [["seeded bread", "white bread"], ["oat milk", "almond milk"], ["bacon", "ham"], ["avocado oil", "olive oil"]]


    # Uncomment block below when you want to use terminal to type in ingredients and replacements 

    # shop._init_markers()

    goal_ingredients = []
    user_input = (input("Enter requested ingredients, separated by commas: ")).lower()
    goal_ingredients = user_input.split(",")

    #algorithm:
    #for items in requested ingredients, make list containing requested ingredient.
    #make list of replacement ingredients.
    # list1 + list 2 = list 3 [requested item, replacement1, replacement2, ...]
    # replacements = [list(1), list(2), list(3), list(4), ...]
    # we want a list containing lists. each list is [requested item, replacement 1, replacement 2, ...]

    temp_list1 = []
    temp2_list2 = []
    temp_replacements = []
    replacements = []

    number_of_ingredients = len(goal_ingredients)
    print(number_of_ingredients)
    for n in range(number_of_ingredients):
        temp_list1 = [goal_ingredients[n]]
        print(temp_list1)
        user_input = (input("Enter replacements for ingredient " + goal_ingredients[n] + " in priority order separated by commas: ")).lower() #creates list of replacements
        temp_list2 = user_input.split(",")
        print(temp_list2)
        if temp_list2 != ['']:
            temp_replacements = temp_list1 + temp_list2
        else:
            temp_replacements = temp_list1
        replacements.append(temp_replacements)
        print(replacements)
    requested_ingredients = goal_ingredients.copy()
    
    #replacements = [["cashews", "almonds", "peanuts"], ["oat milk", "almond milk"], ["bacon", "ham"], ["avocado oil", "olive oil"]]
    #^e.g. if no cashews bring almonds, if no almonds bring peanuts, if no peanuts give up and say not available.


    myRobot = Robot(goal_ingredients, replacements)

    myRobot.send_robot_location(map.states()[myRobot.get_location()])

    print(f"\ngoal ingredients : {myRobot.get_goal_ingredients()}")
        
    # replaces oos items and reports impossible task for oos items without replacements
    bad_ingredients = myRobot.check_for_oos_ingredients(shop, boxes)

    if bad_ingredients != []:  # if any required item is known to be oos and does not have a replacement
        # TODO: if we know that the task is impossible should we still do it? or should we somehow move onto the next customer
        myRobot.report_out_of_stock(bad_ingredients, shop)

    collected_all_items = False


    # get list of states of boxes that the robot needs to visit to pick up all items on the list
    shop.set_boxes_to_visit(myRobot.get_goal_ingredients(), boxes, myRobot)

    if shop.boxes_to_visit == {}:
        print("No ingredients required possible to return.")
        shop.set_path_to_customer()
        collected_all_items = True
    else:
        print(f"goal ingredients after replacements : {myRobot.get_goal_ingredients()} \n")  # prints list of required ingredients after oos items have been replaced

        shop.print_boxes_to_visit()

        # add people and boxes to rviz
        shop.publish_people()
        shop.publish_boxes()
        shop.publish_boxes_to_visit()
        shop.policy_iteration()
        shop.post_start_customer()
    
    i = 1
    number_of_steps_taken = 0
    collected_all_items = False
    
    global steps_until_update
    steps_until_update = 1
    while not myRobot.move_using_policy_iteration(map.states(), shop.pi_transitions, shop.transitions, shop.people):  # until a terminal state is reached
        
        number_of_steps_taken += 1
        
        myRobot.send_robot_location(map.states()[myRobot.get_location()])  # update robot's pose for rviz
        done = False
        
        if myRobot.check_if_at_box(shop.boxes_to_visit.keys()):            # if the robot is at one of the required boxes
            
            if (i == steps_until_update):
                done = True
                shop.set_people(1)
                
                i = 0

                shop.transitions = transitions.transitions(shop.people)
                # print(shop.transitions)
                shop.publish_people()
            
            # get the box at our current state
            for box in boxes:
                if str(box.get_name()) == str(shop.box_states[myRobot.get_location()]):
                    myRobot.current_box = box  
                    break
            
            print("---------------------------------------------------------------------------")    
            print(f"contents of current box : {myRobot.current_box.get_available_ingredients()}")
            
            # picks up all goal ingredients contained within this box


            myRobot.pick_up_all_required_ingredients_from_box(shop)

            # print out robot inventory and contents of box after robot picks up ingredients
            print(f"\nrobot inventory : {myRobot.get_inventory()}")
            print(f"updated box contents: {myRobot.current_box.get_available_ingredients()}")               

            # update boxes to visit in case any new ones were added, e.g. if an item was oos and replacement is in another box
            shop.set_boxes_to_visit(myRobot.get_goal_ingredients(), boxes, myRobot)
            
            shop.update_boxes_to_visit(myRobot.get_location())  # remove box from list of remaining boxes and set its reward back to -1
            shop.print_boxes_to_visit()  # print the list of remaining boxes to visit
                        
            
            if len(shop.boxes_to_visit) == 0:  # if all boxes have been visited then update policy iteration to return to state 8
                shop.publish_boxes_to_visit()
                shop.set_path_to_customer()
                shop.post_end_customer()
                collected_all_items = True

            shop.policy_iteration()

                
        if(i == steps_until_update and done == False):  # TODO: does this always run? is there a better way to implement this?
            # print("change\n\n\n\n")
            shop.set_people(1)
            shop.transitions = transitions.transitions(shop.people)
            i = 0
            shop.policy_iteration()
            shop.publish_people()

        myRobot.send_robot_location(map.states()[myRobot.get_location()])  # send robot location to rviz publisher
            
        i += 1    
        
        if myRobot.get_location() == "s8" and collected_all_items == True:  # if robot is back in state 8 with the customer, end loop
            print("robot now back at customer \n")

            # sort lists alphabetically for readability and then print them 
            requested_ingredients.sort()
            myRobot.get_inventory().sort()
            
            print(f"ingredients requested by customer : {requested_ingredients}")
            print(f"ingredients returned by the robot : {myRobot.get_inventory()}")
            print(f"substitutions : {myRobot.get_substitutions()}")
            print(f"unavailable : {myRobot.get_unavailable()}")
            
            print(f"number of steps taken by robot: {number_of_steps_taken}")

            # TODO: robot actions when it returns to customer
            # do robot inventory processing here
            # e.g. check whether it matches the customer's ingredient list
            # clear inventory?
            # get new customer?            

            break
        
    print("\nsuccess!!!!")
    

if __name__ == '__main__':
    rospy.init_node("map_tester")
    main()
    rospy.spin()

# main()
