import os
import sys
current_directory = os.getcwd()
print("Current working directory:", current_directory)
pythonpath = os.environ.get('PYTHONPATH', '')
print("PYTHONPATH:", pythonpath)
sys.path.append(current_directory)
directories = [current_directory]
pythonpath = os.pathsep.join(directories)
os.environ['PYTHONPATH'] = pythonpath

pythonpath = os.environ.get('PYTHONPATH', '')
print("PYTHONPATH:", pythonpath)

print("System path:")
for path in sys.path:
    print(path)




import random


from RL.rlq.shared.ReplayMemory import ReplayMemory
from collections import deque

        
        
class Memory_Nodes_Stack:
    def __init__(self, parent: 'Memory_Nodes_Stack' = None) -> None:
        self.working_memory = ReplayMemory(4)
        # self.old_memory = None
        self.stack = MyDeQueue()
        self.position_pointers = MyDeQueue()
        self.current_memory = 0
        if parent:
            self.stack = parent.stack
            self.position_pointers = parent.position_pointers
            self.total = parent.total
            self.current_memory = parent.current_memory
            
        else:
            self.total = 0
            self.current_memory = 0
            
        self.update_working_memory_size()
        
    def __len__(self):
        
        return self.working_memory_size + sum(self.position_pointers)
        
    def get_position_pointers(self):
        return (self.working_memory_size, self.position_pointers) 
        
    def at_working_memory(self):
        return self.current_memory == 0
        
        
    def get_memory(self):
        return self.old_memory.get_memory()
    
    def reset_stack(self):
        while len(self.position_pointers) < len(self.stack):
            self.stack.pop()
    
    def old_mem_size(self):
        return len(self.old_memory)
        
    def push(self, *item):
        self.working_memory.push(*item)
        self.update_working_memory_size()
        if self.total + self.working_memory_size > 4:
            self.decrease_pointer()
        
    def make_child(self):
        working_memory = self.working_memory
        self.position_pointers.appendleft(len(working_memory))
        self.total += len(working_memory)
        self.working_memory = ReplayMemory(4)
        self.stack.appendleft(working_memory)
        self.current_memory += 1
        
        
        
    def update_working_memory_size(self):
        self.working_memory_size = self.working_memory.total()
        
    def decrease_pointer(self):
        if not self.position_pointers.is_at_working_memory():
            pointer = self.position_pointers.pop()
        
            if pointer != 1:
                self.position_pointers.append(pointer - 1)
            else:
                self.current_memory -= 1


    def sample(self, sample_size):
        results = []
        if self.at_working_memory():
            return self.working_memory.sample(sample_size)
        else:
            mems = deque()
            for mem in self.stack:
                mems.extend(mem.memory)
            mems.extend(self.working_memory.memory)
            return random.sample(mems, sample_size)
    
class MyDeQueue(deque):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    

    def is_at_working_memory(self):
        return len(self) == 0
                            
class Stack:
    def __init__(self):
        self.items = []

    def is_at_working_memory(self):
        return len(self.items) == 1
    
    def is_at_old_memory(self):
        return len(self.items) == 2

    def is_empty(self):
        return len(self.items) == 0
    
    def get_current_memory(self):
        return self.items[-1]
    
    def get_working_memory(self):
        return self.items[0]
        
    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            raise IndexError("cannot pop further")

    def size(self):
        return len(self.items)
    
    
    
    
    

mem1 = Memory_Nodes_Stack()
mem1.push("a","a","a","a")
mem1.push("b","b","b","b")
mem1.push("c","c","c","c")
mem1.push("d","d","d","d")
print(mem1.working_memory.memory)
print(mem1.stack)

mem1.make_child()
mem2 = Memory_Nodes_Stack(mem1)

print("\n")
print(mem1.working_memory.memory)
print(mem1.stack[0].memory)

mem1.make_child()
mem3 = Memory_Nodes_Stack(mem1)

print("\n")
print(mem1.working_memory.memory)
print(mem1.stack[0].memory)
print(mem1.stack[1].memory)

mem1.push(5,5,5,5)
print("\n")
print(mem1.working_memory.memory)
print(mem1.stack[0].memory)
print(mem1.stack[1].memory)

print("\n")
print(mem1.sample(4))

print("\n")
print(mem3.working_memory.memory)
print(mem3.stack[0].memory)
print(mem3.stack[1].memory)

