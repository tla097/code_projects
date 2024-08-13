class Customer:
    def __init__(self, recipe, replacements):
        self._recipe = recipe
        self.replacements = replacements
        self.ingredients = []

    def request_recipe(self):
        return self._recipe

    def receive_ingredients(self, ingredients):
        self.ingredients = ingredients
        self.ingredients.sort()

    def is_correct_result(self):
        temp = self._recipe
        temp.sort()

        for i in range (len(self.ingredients)):
            if self._recipe[i] != self.ingredients[i]:
                print(f"ingredient {self.ingredients[i]} is wrong. ")
                return False
        return True
