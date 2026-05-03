#4-slot Pool ACM
import threading

#libraries

#ACM Class

class ACM():
    def __init__(self):
            self.s = [None]*3 # set by the writer and are visible by the reader
            self.w = 0
            self.r = 1 # visible to the writer
            self.l = 0 #latest data : set by the writer and are visible by the reader
           
    def read(self):   
            self.r = self.l # r takes the slot of l
            data = self.s[self.r] #reads slot
            return data
        
    def write(self ,data):
            #looks for an available slot
            slot_available = False
            while slot_available == False: #stops when it finds available slot
                for n in range (3):
                    if n != self.r and n != self.l: #avoids the slot being used by the reader and not use the previous slot
                        self.w = n
                        slot_available = True 
            self.s[self.w] = data
            #print(f"Writing '{data}' into slot {self.s}")
            self.l = self.w
            slot_available = False


            
            
       
                
                
                