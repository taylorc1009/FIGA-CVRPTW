import copy
from destination import Destination
from typing import List, Dict, Union
from problemInstance import ProblemInstance
from node import Node

class Vehicle:
    def __init__(self, current_capacity: int=0, destinations: List[Destination]=None, route_distance: float=0.0) -> None:
        self.current_capacity: int=int(current_capacity)
        self.destinations: List[Destination]=destinations
        self.route_distance: float=float(route_distance)

    def __str__(self) -> str:
        return f"capacity={self.current_capacity}, distance={self.route_distance}, {len(self.destinations)=}, {[d.node.number for d in self.destinations]})"

    def get_customers_visited(self) -> List[Destination]:
        return self.destinations[1:-1] # to do this, we assume that every depot departure and return is ordered correctly (at index 0 and n - 1) which we can do as any route that isn't in this format is incorrect

    def get_num_of_customers_visited(self) -> int:
        return len(self.destinations) - 2 # like "get_customers_visited", to do this, we assume that every depot departure and return is ordered correctly

    def calculate_destination_time_window(self, instance: ProblemInstance, previous_destination: int, current_destination: int) -> bool:
        previous_destination = self.destinations[previous_destination]
        current_destination = self.destinations[current_destination]
        current_destination.arrival_time = previous_destination.departure_time + instance.get_distance(previous_destination.node.number, current_destination.node.number)
        if current_destination.arrival_time < current_destination.node.ready_time: # if the vehicle arrives before "ready_time" then it will have to wait for that moment before serving the node
            current_destination.wait_time = current_destination.node.ready_time - current_destination.arrival_time
            current_destination.arrival_time = current_destination.node.ready_time
        else:
            current_destination.wait_time = 0.0
        current_destination.departure_time = current_destination.arrival_time + current_destination.node.service_duration
        return current_destination.arrival_time <= current_destination.node.due_date

    def calculate_destinations_time_windows(self, instance: ProblemInstance, start_from: int=1) -> bool:
        is_feasible_route = True
        for i in range(start_from, len(self.destinations)):
            if not self.calculate_destination_time_window(instance, i - 1, i):
                is_feasible_route = False
        return is_feasible_route

    def calculate_vehicle_load(self) -> None:
        self.current_capacity = sum(destination.node.demand for destination in self.get_customers_visited())

    def calculate_length_of_route(self, instance: ProblemInstance) -> None:
        self.route_distance = sum(instance.get_distance(self.destinations[i - 1].node.number, self.destinations[i].node.number) for i in range(1, len(self.destinations)))

    def __deepcopy__(self, memodict: Dict=None) -> "Vehicle":
        return Vehicle(current_capacity=self.current_capacity, route_distance=self.route_distance, destinations=[copy.deepcopy(d) for d in self.destinations])

    @classmethod
    def create_route(cls, instance: ProblemInstance, node: Union[Node, Destination, List[Node], List[Destination]]=None) -> "Vehicle":
        if node:
            if isinstance(node, list):
                if isinstance(node[0], Node):
                    return cls(destinations=[Destination(node=instance.nodes[0]), *[Destination(node=n) for n in node], Destination(instance.nodes[0])])
                elif isinstance(node[0], Destination):
                    return cls(destinations=[Destination(node=instance.nodes[0]), *[Destination(node=d.node) for d in node], Destination(node=instance.nodes[0])])
            elif isinstance(node, Node):
                return cls(destinations=[Destination(node=instance.nodes[0]), Destination(node=node), Destination(instance.nodes[0])])
            elif isinstance(node, Destination):
                return cls(destinations=[Destination(node=instance.nodes[0]), Destination(node=node.node), Destination(instance.nodes[0])])
        else:
            return cls(destinations=[Destination(node=instance.nodes[0]), Destination(instance.nodes[0])])
