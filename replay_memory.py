#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 19:15:37 2021

@author: josev
"""

from collections import deque, namedtuple
import random


class experience_replay:
    
    def __init__(self, queue_capacity=500000):
        self.queue_capacity = queue_capacity
        self.memory = deque(maxlen=queue_capacity)

    def add_to_memory(self, experience_tuple):
        self.memory.append(experience_tuple)
        

    def get_batch_for_replay(self, batch_size=64):
        try:
            return random.sample(self.memory, batch_size)
        except ValueError:
            return None

    def get_memory_size(self):
        return len(self.memory)

    def delete_memory(self):
        self.memory = deque(maxlen=self.queue_capacity)
         
if __name__ == "__main__":
    
    Transition = namedtuple("Transition", ('state', 'action', 'reward', 'next_state', "done"))
    
    rm = experience_replay()
    rm.add_to_memory((1, 2, 3, 4, 5))
    rm.add_to_memory((6, 7, 8, 9, 10))
    batch = rm.get_batch_for_replay(2)
    print(Transition(*zip(*batch)))
    
    print(rm.get_memory_size())
    
    