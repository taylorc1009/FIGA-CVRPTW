from typing import Dict, Union
from math import sqrt

class Node:
    def __init__(self, number: int, x: int, y: int, demand: int, ready_time: int, due_date: int, service_duration: int) -> None:
        self.number: int=int(number)
        self.x: int=int(x)
        self.y: int=int(y)
        self.demand: int=int(demand)
        self.ready_time: int=int(ready_time)
        self.due_date: int=int(due_date)
        self.service_duration: int=int(service_duration)

    def get_distance(self, *args: Union["Node", int]) -> float:
        xPow, yPow = None, None
        if len(args) == 1 and type(args[0]) is type(self):
            xPow, yPow = (args[0].x - self.x) ** 2, (args[0].y - self.y) ** 2
        elif len(args) == 2 and type(args[0]) is int and type(args[1]) is int:
            xPow, yPow = (args[0] - self.x) ** 2, (args[1] - self.y) ** 2
        return sqrt(xPow + yPow)
    
    def __str__(self) -> str:
        return f"Node(number={self.number}, x={self.x}, y={self.y}, demand={self.demand}, ready_time={self.ready_time}, due_date={self.due_date}, service_duration={self.service_duration})"

    def __deepcopy__(self, memodict: Dict=None) -> "Node":
        return Node(number=self.number, x=self.x, y=self.y, demand=self.demand, ready_time=self.ready_time, due_date=self.due_date, service_duration=self.service_duration)
