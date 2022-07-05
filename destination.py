from typing import Dict
from node import Node

class Destination:
    def __init__(self, node: Node=None, arrival_time: float=0.0, departure_time: float=0.0, wait_time: float=0.0) -> None:
        self.node: Node=node
        self.arrival_time: float=float(arrival_time)
        self.departure_time: float=float(departure_time)
        self.wait_time: float=float(wait_time)

    def __str__(self) -> str:
        return f"arrival_time={self.arrival_time}, departure_time={self.departure_time}, wait_time={self.wait_time}, {str(self.node)}"

    def __deepcopy__(self, memodict: Dict=None) -> "Destination": # self.node doesn't need to be deep-copied as the nodes should never be modified, so they can be given by reference
        return Destination(node=self.node, arrival_time=self.arrival_time, departure_time=self.departure_time, wait_time=self.wait_time)
