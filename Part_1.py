import threading
import time
import random

random.seed(42) ### Apply random seed to ensure reproducibility

class Buffer(threading.Thread):
    def __init__(self , size):
        super().__init__()  
        self.size = size
        self.buffer = [None for _ in range(size)]
        self.in_index = 0
        self.out_index = 0
        self.mutex = threading.Semaphore(1)
        self.empty = threading.Semaphore(size)
        self.full = threading.Semaphore(0)
    
    def consume(self):
        items_consumed = 0
        while items_consumed < self.size: # Similarly, temporary condition so the Consumer does not run forever more
            self.full.acquire()
            self.mutex.acquire()

            data = self.buffer[self.out_index]
            self.buffer[self.out_index] = None
            #print(f"Consumer {self.name} consumed'{data}' from memory location {self.out_index}. Current state of buffer: {self.buffer}")
            self.out_index = (self.out_index + 1) % self.size
            items_consumed += 1
            
            self.mutex.release()
            self.empty.release()     
        return data
    
    def produce(self,data):
        items_produced = 0
        
        while items_produced < self.size: 
            self.empty.acquire()
            self.mutex.acquire()
            
            self.buffer[self.in_index] = data
            #print(f"Producer {self.name} produced data '{data}' and stored in memory location {self.in_index}. Current state of buffer: {self.buffer}")
            self.in_index = (self.in_index + 1) % self.size #circular buffer
            items_produced += 1
            
            self.mutex.release()
            self.full.release()





# Testing of Buffer
'''
if __name__ == "__main__":
    start = time.time()

    buffer = Buffer(size = 5)

    buffer.start()
    buffer.join()

    end = time.time()
    print(f"Total runtime: {end - start:.2f} seconds")
'''


