import copy
from typing import List, Dict
from MMOEASA.constants import INFINITY
from common import INT_MAX
from vehicle import Vehicle
from problemInstance import ProblemInstance
from solution import Solution

class MMOEASASolution(Solution):
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, default_temperature: float=0.0, temperature: float=0.0, cooling_rate: float=0.0, total_distance: float=0.0, distance_unbalance: float=0.0, cargo_unbalance: float=0.0, rank: int=INT_MAX) -> None:
        super(MMOEASASolution, self).__init__(_id=_id, vehicles=vehicles, feasible=feasible, total_distance=total_distance, rank=rank)
        self.distance_unbalance: float=float(distance_unbalance)
        self.cargo_unbalance: float=float(cargo_unbalance)
        self.default_temperature: float=float(default_temperature)
        self.temperature: float=float(temperature)
        self.cooling_rate: float=float(cooling_rate)

    def __str__(self) -> str:
        return f"total_distance={self.total_distance}, distance_unbalance={self.distance_unbalance}, cargo_unbalance={self.cargo_unbalance}, {len(self.vehicles)=}, {[f'{i}. {str(v)}' for i, v in enumerate(self.vehicles)]}"

    def objective_function(self, instance: ProblemInstance) -> None:
        vehicle = 0
        self.total_distance = 0.0
        self.feasible = True # set the solution as feasible temporarily

        while vehicle < len(self.vehicles) and self.feasible:
            self.total_distance += self.vehicles[vehicle].route_distance

            for destination in self.vehicles[vehicle].get_customers_visited():
                if destination.arrival_time > instance.nodes[destination.node.number].due_date or self.vehicles[vehicle].current_capacity > instance.capacity_of_vehicles:
                    self.feasible = False
                    self.total_distance = INFINITY
                    self.distance_unbalance = INFINITY
                    self.cargo_unbalance = INFINITY
                    break
            vehicle += 1

        if self.feasible:
            minimum_distance = INFINITY
            maximum_distance = 0
            minimum_cargo = INFINITY
            maximum_cargo = 0

            for vehicle in self.vehicles:
                # these cannot be converted to "if ... elif" because we may miss, for example, our "maximum_distance" as on the first iteration it will also be less than "INFINITY" ("minimum_distance")
                if vehicle.route_distance < minimum_distance:
                    minimum_distance = vehicle.route_distance
                if vehicle.route_distance > maximum_distance:
                    maximum_distance = vehicle.route_distance
                if vehicle.current_capacity < minimum_cargo:
                    minimum_cargo = vehicle.current_capacity
                if vehicle.current_capacity > maximum_cargo:
                    maximum_cargo = vehicle.current_capacity
            self.distance_unbalance = maximum_distance - minimum_distance
            self.cargo_unbalance = maximum_cargo - minimum_cargo

    def __deepcopy__(self, memodict: Dict=None) -> "MMOEASASolution":
        return MMOEASASolution(_id=self.id, vehicles=[copy.deepcopy(v) for v in self.vehicles], feasible=self.feasible, default_temperature=self.default_temperature, temperature=self.temperature, cooling_rate=self.cooling_rate, total_distance=self.total_distance, distance_unbalance=self.distance_unbalance, cargo_unbalance=self.cargo_unbalance, rank=self.rank)
