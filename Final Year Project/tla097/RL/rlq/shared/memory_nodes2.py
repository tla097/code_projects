import random
from RL.rlq.shared.ReplayMemory import ReplayMemory

class Memory_Nodes:
    def __init__(self, value):
        self.memory_to_add_to = ReplayMemory(10000)
        self.mem_size = len(self.memory_to_add_to)
        self.old_memory = None
        self.prev = None
        self.memory_pointer = 0
        self.position_pointer = 0
        
        
    def decrease_pointer(self):
        if self.memory_pointer != 0:
            self.position_pointer -=1
            if self.position_pointer == 0:
                self.memory_pointer -= 1
                if self.memory_pointer <= 1:
                    if self.prev.memory_pointer == 0:
                        self.prev = None
                        
    def push(self, *args):
        self.memory_to_add_to.push(args)
        self.mem_size += 1
        
    def made_child(self):
        self.old_memory = self.memory_to_add_to
        self.memory_to_add_to = ReplayMemory(10000)
        self.mem_size = 0
        
    def sample(self, sample_size):
        result = []
        for i in range(sample_size):
            if self.memory_pointer == 0:
                result.append(self.memory_to_add_to.sample(sample_size))

            if self.memory_pointer == 1:
                number = random.randint(1, self.position_pointer + self.mem_size)
                if number >= self.position_pointer:
                    result.append(self.memory_to_add_to.sample(1))
                else:
                    ran = random.randint(len(self.old_memory) - 1- pos ,len(self.old_memory) -1)
                    result.append(self.old_memory[ran])
                self.decrease_pointer()
                
            
            if self.memory_pointer > 1:
                sizes = []
                sizes.append(self.mem_size)
                sizes.append(len(self.old_memory))
                for i in range(2, self.memory_pointer + 1):
                    pass
                    
                    # plan
                    # make an array of sizes of the previous memories
                    # do the random sampling of the previous memories
                    # test it
                    # implement it into the evolutionary runner page
                    # run it and fingers crossed
                    
                    
            
                
                
            
            
            
mem = ReplayMemory(1000)

mem.push(1,2,3,4)

mem.push(5, 6,7,8)

mem.push(9, 10,11,12)

pos = 1

ran = random.randint(len(mem.memory) - 1- pos ,len(mem.memory) -1)
print(mem.memory[ran])

for i in range(2,2):
    print(i)
            
                
            
                        
                        
    
            
            
            
