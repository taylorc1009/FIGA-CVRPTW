from typing import List
from vehicle import Vehicle
from problemInstance import ProblemInstance
from common import INT_MAX

class Solution:
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, total_distance: float=0.0, rank: int=INT_MAX) -> None:
        self.id: int=int(_id)
        self.vehicles: List[Vehicle]=vehicles
        self.feasible: bool=feasible
        self.total_distance: float=float(total_distance)

        # ranking used in Ombuki's Algorithm; exists in the parent class for when Ombuki's Algorithm is solving MMOEASA's objective function
        self.rank: int=int(rank)

    def calculate_nodes_time_windows(self, instance: ProblemInstance) -> None:
        for i, _ in enumerate(self.vehicles):
            self.vehicles[i].calculate_destinations_time_windows(instance)

    def calculate_vehicles_loads(self, instance: ProblemInstance) -> None:
        for i, _ in enumerate(self.vehicles):
            self.vehicles[i].calculate_vehicle_load(instance)

    def calculate_length_of_routes(self, instance: ProblemInstance) -> None:
        for i, _ in enumerate(self.vehicles):
            self.vehicles[i].calculate_length_of_route(instance)

    def check_format_is_correct(self, instance: ProblemInstance) -> None:
        # error checks to ensure that every route is of the valid format
        if sum([v.get_num_of_customers_visited() for v in self.vehicles]) != instance.amount_of_vehicles - 1: # check if the solution contains the correct amount of destinations, in that it visits all of them
            raise ValueError(f"Mismatched amount of destinations: {sum([v.get_num_of_customers_visited() for v in self.vehicles])}")
        elif [v for v in self.vehicles if len(v.destinations) < 2]: # checks if all the routes has at least 2 destinations; routes should always at least depart from and return to the depot
            raise ValueError("Number of destinations was not at least 2")
        elif [v for v in self.vehicles if v.destinations[0].node.number or v.destinations[-1].node.number]: # checks that every route starts and ends at the depot
            raise ValueError("Indexes 0 and n - 1 should be depot nodes")
        elif [set(range(1, 101)).remove(d.node.number) for v in self.vehicles for d in v.get_customers_visited()][0]: # checks if all nodes have been visited; ".remove" will also find both: duplicate nodes as it will throw an exception when it tries to remove an already-removed node, and depot nodes in the middle of a route as the set starts at 1, so if it tries to remove the depot (node 0) it won't exist and throw an exception
            raise ValueError("Not all nodes are visited")
