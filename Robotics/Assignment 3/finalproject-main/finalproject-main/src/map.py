#!/usr/bin/env python3

    
def grid_map():
    grid_map = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
    [100, 0, 0, 0, 100, 0, 0, 0, 0],
    [100, 0, 0, 0, 0, 0, 100, 0, 100],
    [100, 0, 0, 0, 0, 0, 0, 0, 100],
    [0, 0, 0, 0, 100, 0, 100, 0, 100],
    [100, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 100, 0, 0, 0, 100],
    [100, 0, 100, 0, 0, 0, 100, 0, 0],
    [0, 0, 0, 0, 100, 0, 0, 0, 100],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 100, 0, 100, 0, 100, 100, 0, 0]]
    return grid_map

def boxes():
    # Now I guess we don't use this
    return boxes

def states():
    states = {'s0': (0, 0), 's1': (1, 0), 's2': (2, 0), 's3': (3, 0), 's4': (4, 0), 's5': (5, 0), 's6': (6, 0), 's7': (7, 0),
            's8': (8, 0), 's9': (1, 1), 's10': (2, 1), 's11': (3, 1), 's12': (5, 1), 's13': (6, 1), 's14': (7, 1),
            's15': (8, 1), 's16': (1, 2), 's17': (2, 2), 's18': (3, 2), 's19': (4, 2), 's20': (5, 2), 's21': (7, 2),
            's22': (1, 3), 's23': (2, 3), 's24': (3, 3), 's25': (4, 3), 's26': (5, 3), 's27': (6, 3), 's28': (7, 3),
            's29': (0, 4), 's30': (1, 4), 's31': (2, 4), 's32': (3, 4), 's33': (5, 4), 's34': (7, 4), 's35': (1, 5),
            's36': (2, 5), 's37': (3, 5), 's38': (4, 5), 's39': (5, 5), 's40': (6, 5), 's41': (7, 5), 's42': (8, 5),
            's43': (0, 6), 's44': (1, 6), 's45': (2, 6), 's46': (3, 6), 's47': (5, 6), 's48': (6, 6), 's49': (7, 6),
            's50': (1, 7), 's51': (3, 7), 's52': (4, 7), 's53': (5, 7), 's54': (7, 7), 's55': (8, 7), 's56': (0, 8),
            's57': (1, 8), 's58': (2, 8), 's59': (3, 8), 's60': (5, 8), 's61': (6, 8), 's62': (7, 8), 's63': (0, 9),
            's64': (1, 9), 's65': (2, 9), 's66': (3, 9), 's67': (4, 9), 's68': (5, 9), 's69': (6, 9), 's70': (7, 9),
            's71': (8, 9), 's72': (0, 10), 's73': (2, 10), 's74': (4, 10), 's75': (7, 10), 's76': (8, 10)}
    return states
    
def actions():
    actions = ("UP", "DOWN", "LEFT", "RIGHT", "TERMINAL")
    return actions


def possible_transitions():
    
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TERMINAL = "TERMINAL"

    possible_transitions = {
            # "s0" : [TERMINAL],## changes here to add terminal states
            "s0": [RIGHT],
            "s1" : [LEFT, RIGHT, UP],
            "s2" : [LEFT, RIGHT],
            "s3" : [LEFT, RIGHT, UP],
            "s4" : [LEFT, RIGHT],
            "s5" : [LEFT, RIGHT, UP],
            "s6" : [LEFT, RIGHT],
            # "s6": [TERMINAL],  # changes here to add terminal states
            "s7" : [LEFT, RIGHT, UP],
            "s8" : [TERMINAL],
            # "s8" : [LEFT, RIGHT, UP],
            "s9" : [RIGHT, UP, DOWN],
            "s10" : [LEFT],
            "s11" : [UP, DOWN],
            "s12" : [RIGHT, UP, DOWN],
            # "s12": [TERMINAL],
            "s13" : [LEFT],
            "s14" : [RIGHT, UP, DOWN],
            "s15" : [LEFT, DOWN],
            "s16" : [UP, DOWN],
            # "s16" : [TERMINAL],
            "s17" : [RIGHT],
            "s18" : [LEFT, UP, DOWN],
            "s19" : [RIGHT],
            "s20" : [LEFT, UP, DOWN],
            "s21" : [UP, DOWN],
            "s22" : [RIGHT, UP, DOWN],
            "s23" : [LEFT],
            "s24" : [RIGHT, UP, DOWN],
            "s25" : [LEFT],
            "s26" : [UP, DOWN],
            "s27" : [RIGHT],
            "s28" : [LEFT, UP, DOWN],
            "s29" : [RIGHT],
            "s30" : [LEFT, UP, DOWN],
            "s31" : [RIGHT],
            "s32" : [LEFT, UP, DOWN],
            "s33" : [UP, DOWN],
            "s34" : [UP, DOWN],
            "s35" : [RIGHT,UP,DOWN],
            "s36" : [RIGHT,LEFT,UP],
            "s37" : [RIGHT,LEFT,UP,DOWN],
            "s38" : [RIGHT, LEFT],
            "s39" : [RIGHT, LEFT, UP, DOWN],
            "s40" : [RIGHT,LEFT, UP],
            "s41" : [RIGHT, LEFT, UP, DOWN],
            "s42" : [LEFT],
            "s43" : [RIGHT],
            "s44" : [LEFT, UP, DOWN],
            "s45" : [DOWN],
            "s46" : [UP, DOWN],
            "s47" : [UP, DOWN],
            "s48" : [DOWN],
            "s49" : [UP, DOWN],
            "s50" : [UP, DOWN],
            "s51" : [RIGHT, UP, DOWN],
            "s52" : [LEFT],
            "s53" : [UP, DOWN],
            "s54" : [RIGHT, UP, DOWN],
            "s55" : [LEFT],
            "s56" : [RIGHT],
            "s57" : [LEFT, UP, DOWN],
            "s58" : [UP],
            "s59" : [UP, DOWN],
            "s60" : [UP, DOWN],
            "s61" : [UP],
            "s62" : [UP, DOWN],
            "s63" : [RIGHT, UP],
            "s64" : [RIGHT, LEFT, DOWN],
            "s65" : [RIGHT, LEFT, UP, DOWN],
            "s66" : [RIGHT, LEFT, DOWN],
            "s67" : [RIGHT, LEFT, UP],
            "s68" : [RIGHT, LEFT, DOWN],
            "s69" : [RIGHT, LEFT, DOWN],
            "s70" : [RIGHT, LEFT, UP, DOWN],
            "s71" : [LEFT, UP],
            "s72" : [DOWN],
            "s73" : [DOWN],
            "s74" : [DOWN],
            "s75" : [DOWN],
            "s76" : [DOWN]
        }
    return possible_transitions



    def shrink_map(self):
        """
        :return: generates new map made from the centers of our cells
        """
        # y = 0
        # new_map = [[]]
        # should_be = [[0, 100, 0, 100, 0, 100, 100, 0, 0],
        #                   [0, 0, 0, 0, 0, 0,0,0,0],
        #                   [0,0,0,0,100,0,0,0,100],
        #                   [100,0,100,0,0,0,100,0],
        #                   [0,0,0,0,100,0,0,0,100],
        #                   [100,0,0,0,0,0,0,0,0],
        #                   [0,0,0,0,100,0,100,0,100],
        #                   [100,0,0,0,0,0,0,0,100],
        #                   [100,0,0,0,0,0,100,0,100],
        #                   [100,0,0,0,100,0,0,0,0],
        #                   [0,0,0,0,0,0,0,0,0,0]]
        #
        map_total = self._map.info.height * self._map.info.width
        new_map = []

        new2d = []


        for row in range(3, self._map.info.height, 7):
            for col in range(3, self._map.info.width, 7):
                new2d.append(self._map_data[col + row * 63])


        for row1 in range(11):
            temp= []
            for col1 in range(9):
                temp.append(new2d[col1 + row1 * 9])
            new_map.append(temp)

        return new_map
