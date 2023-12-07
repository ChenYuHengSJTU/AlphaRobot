import math,heapq
import time

class timer:
    def __init__(self) -> None:
        self.start_time=None
    def start(self):
        self.start_time=time.perf_counter()
    def check(self):
        return time.perf_counter()-self.start_time
        
NUM_DATA=299
total=0
maximum=-1
minimum=100

with open("err rate.txt","r") as f:
    for i in range(NUM_DATA):
        s=f.readline()
        tmp=float(s)
        total+=tmp
        maximum=max(maximum,tmp)
        minimum=min(minimum,tmp)

avg=total/NUM_DATA
print(f"[data]: \tavg={avg}")

total=0
with open("A_performance.txt","r") as f:
    for i in range(NUM_DATA):
        s=(float(f.readline())-avg)**2
        total+=s

covar=math.sqrt(total/NUM_DATA)

print(f"[data]: \tcovar={covar}")
print(f"[data]: \tmax={maximum}, min={minimum}")


