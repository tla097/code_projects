class Box:
    # form: Box("Dairy", {"milk": 3, "oat milk": 1, "almond milk": 1})
    # all possible box ingredients are included in dictionary, but OOS items will have quantity 0

    def __init__(self, name, available_ingredients):
        # name is a string of the form: "Dairy"
        # available ingredients is a dictionary of items and quantities of the form:
        # {"milk": 3, "oat milk": 1, "almond milk": 1}
        self._name = name
        self._available_ingredients = available_ingredients

    def set_ingredients(self, ingredients): # used to initially set the ingredients in a box along with quantities of each
        self._available_ingredients = ingredients

    def check_for_ingredient(self, ingredient):
        return (ingredient in self._available_ingredients.keys()  # if ingredient exists in box (0+ items)
                and self._available_ingredients[ingredient] > 0)  # and is in stock (1+ items) return True

    def put_back_ingredient(self, ingredient):
        so_far = self._available_ingredients[ingredient]
        self._available_ingredients.update({ingredient : so_far + 1})

    def get_quantity_of_ingredient(self, ingredient):
        return self._available_ingredients[ingredient]

    def valid_ingredient(self, ingredient):
        return ingredient in self._available_ingredients.keys()  # if ingredient is one of the possible ingredients in this box return True

    def retrieve_ingredient(self, ingredient):  # remove ingredient from box
        if self.check_for_ingredient(ingredient):  # if ingredient is in stock
            self._available_ingredients[ingredient] -= 1  # subtract one from quantity of item available
            return ingredient  # return the requested ingredient
        return -1  # if ingredient is out of stock/ not in this box then return -1

    def add_ingredient(self, ingredient):  # add ingredient to box
        if self.valid_ingredient(ingredient):  # if we can put the ingredient in the box
            self._available_ingredients[ingredient] += 1  # increment item quantity by one
        else:
            # if ingredient cannot be added, don't do anything
            print("*tried to add invalid item to box*")

    def is_empty(self):  # check if box is empty
        return all(v == 0 for v in self._available_ingredients.values())  # if all item quantities are 0, i.e. box is empty

    def get_name(self):
        return self._name

    def get_available_ingredients(self):
        return self._available_ingredients

    def __str__(self):
        return self._name + ""

# shop contains boxes,
# boxes contain ingredients
# boxes = what it can carry
# what it does carry
