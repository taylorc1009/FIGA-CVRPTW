from typing import List
from vehicle import Vehicle
from problemInstance import ProblemInstance
from common import INT_MAX

class Solution:
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, total_distance: float=0.0, rank: int=INT_MAX, default_temperature: float=0.0, temperature: float=0.0, cooling_rate: float=0.0) -> None:
        self.id: int=int(_id)
        self.vehicles: List[Vehicle]=vehicles
        self.feasible: bool=feasible
        self.total_distance: float=float(total_distance)

        # Simulated Annealing parameters for MMOEASA: exists in the parent class for when MMOEASA is solving Ombuki's objective function
        self.default_temperature: float=float(default_temperature)
        self.temperature: float=float(temperature)
        self.cooling_rate: float=float(cooling_rate)

        # ranking used in Ombuki's Algorithm; exists in the parent class for when Ombuki's Algorithm is solving MMOEASA's objective function
        self.rank: int=int(rank)

    def calculate_routes_time_windows(self, instance: ProblemInstance) -> None:
        for i, _ in enumerate(self.vehicles):
            self.vehicles[i].calculate_destinations_time_windows(instance)

    def calculate_vehicles_loads(self) -> None:
        for i, _ in enumerate(self.vehicles):
            self.vehicles[i].calculate_vehicle_load()

    def calculate_length_of_routes(self, instance: ProblemInstance) -> None:
        for i, _ in enumerate(self.vehicles):
            self.vehicles[i].calculate_length_of_route(instance)

    def check_format_is_correct(self, instance: ProblemInstance) -> None:
        node_nums = set(range(1, 101))
        # error checks to ensure that every route is of the valid format
        if sum(v.get_num_of_customers_visited() for v in self.vehicles) != len(instance.nodes) - 1: # check if the solution contains the correct amount of destinations, in that it visits all of them
            raise ValueError(f"Mismatched amount of destinations: {sum(v.get_num_of_customers_visited() for v in self.vehicles)}")
        elif [v for v in self.vehicles if len(v.destinations) < 3]: # checks if all the routes have at least 3 destinations; routes should always at least depart from and return to the depot, while visiting a customer inbetween
            raise ValueError("Number of destinations was not at least 3")
        elif [v for v in self.vehicles if v.destinations[0].node.number or v.destinations[-1].node.number]: # checks that every route starts and ends at the depot
            raise ValueError("Indexes 0 and n - 1 should be depot nodes")
        elif node_nums.difference(set(d.node.number for v in self.vehicles for d in v.get_customers_visited())): # checks if all nodes have been visited; ".remove" will also find both: duplicate nodes as it will throw an exception when it tries to remove an already-removed node, and depot nodes in the middle of a route as the set starts at 1, so if it tries to remove the depot (node 0) it won't exist and throw an exception
            raise ValueError("Not all nodes are visited")
