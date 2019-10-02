# -*- coding: utf-8 -*-
#Slow bot implementation based on "Deep Learning and the Game of Go" chapter 3

import copy
from gotypes import Player


#There are three basic kinds of moves in go:
#Making a play
#Passing a turn
#Resigning the game
class Move():
    def __init__(self, point = None, is_pass = False, is_resign = False):
        #ensures a given move is only one of play, pass, or resign
        assert (point is not None) ^ is_pass ^ is_resign    
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign
    
    @classmethod
    def play(cls, point):
        return Move(point = point)
        
    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)
        
    @classmethod
    def resign(cls):
        return Move(is_resign=True)
    

#Data type to track strings of stones of the same color, logic is not here!
class GoString():
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)
        
    def remove_liberty(self, point):
        self.liberties.remove(point)
        
    def add_liberty(self, point):
        self.liberties.add(point)
        
    def merged_with(self, go_string):
        assert go_string.color == self.color
        #this is really neat syntax for sets...
        combined_stones = self.stones | go_string.stones
        return GoString(
                self.color,
                combined_stones,
                (self.liberties | go_string.liberties) - combined_stones)
        
    @property
    def num_liberties(self):
        return len(self.liberties)
    
    def __eq__(self,other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties