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

    def calculate_destination_time_window(self, instance: ProblemInstance, previous_destination: int, current_destination: int) -> None:
        previous_node = self.destinations[previous_destination].node.number
        current_node = self.destinations[current_destination].node.number
        self.destinations[current_destination].arrival_time = self.destinations[previous_destination].departure_time + instance.get_distance(previous_node, current_node)
        if self.destinations[current_destination].arrival_time < instance.nodes[current_node].ready_time:  # if the vehicle arrives before "ready_time" then it will have to wait for that moment before serving the node
            self.destinations[current_destination].wait_time = instance.nodes[current_node].ready_time - self.destinations[current_destination].arrival_time
            self.destinations[current_destination].arrival_time = instance.nodes[current_node].ready_time
        else:
            self.destinations[current_destination].wait_time = 0.0
        self.destinations[current_destination].departure_time = self.destinations[current_destination].arrival_time + instance.nodes[current_node].service_duration

    def calculate_destinations_time_windows(self, instance: ProblemInstance) -> None:
        for i in range(1, len(self.destinations)):
            self.calculate_destination_time_window(instance, i - 1, i)

    def calculate_vehicle_load(self, instance: ProblemInstance) -> None:
        self.current_capacity = sum([instance.nodes[self.destinations[i].node.number].demand for i in range(1, len(self.destinations) - 1)])

    def calculate_length_of_route(self, instance: ProblemInstance) -> None:
        self.route_distance = sum([instance.get_distance(self.destinations[i - 1].node.number, self.destinations[i].node.number) for i in range(1, len(self.destinations))])

    def __deepcopy__(self, memodict: Dict=None) -> "Vehicle":
        return Vehicle(current_capacity=self.current_capacity, route_distance=self.route_distance, destinations=[copy.deepcopy(d) for d in self.destinations])

    @classmethod
    def create_route(cls, instance: ProblemInstance, node: Union[Node, List[Node], List[Destination]]=None) -> "Vehicle":
        if node:
            if isinstance(node, list):
                if isinstance(node[0], Node):
                    destinations = [Destination(node=n) for n in node]
                    return cls(destinations=[Destination(node=instance.nodes[0]), *destinations, Destination(instance.nodes[0])])
                elif isinstance(node[0], Destination):
                    return cls(destinations=[Destination(node=instance.nodes[0]), *node, Destination(node=instance.nodes[0])])
            elif isinstance(node, Node):
                return cls(destinations=[Destination(node=instance.nodes[0]), Destination(node=node), Destination(instance.nodes[0])])
        else:
            return cls(destinations=[Destination(node=instance.nodes[0]), Destination(instance.nodes[0])])
