import copy
from typing import List, Dict
from common import INT_MAX
from vehicle import Vehicle
from problemInstance import ProblemInstance
from solution import Solution

class FIGASolution(Solution):
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, total_distance: float=0.0, num_vehicles: int=0) -> None:
        super(FIGASolution, self).__init__(_id=_id, vehicles=vehicles, feasible=feasible, total_distance=total_distance)
        self.num_vehicles: int=int(num_vehicles) # the reason this objective is a variable instead of just using "len(vehicles)" is because if the solution is invalid, it needs to be set to a very high number

    def __str__(self) -> str:
        return f"total_distance={self.total_distance}, num_vehicles={self.num_vehicles}, {len(self.vehicles)=}, {[f'{i}. {str(v)}' for i, v in enumerate(self.vehicles)]}"

    def objective_function(self, instance: ProblemInstance) -> None:
        self.total_distance = 0.0
        self.num_vehicles = len(self.vehicles)
        self.feasible = True  # set the solution as feasible temporarily

        for vehicle in self.vehicles:
            self.total_distance += vehicle.route_distance

            for destination in vehicle.get_customers_visited():
                if destination.arrival_time > instance.nodes[destination.node.number].due_date or vehicle.current_capacity > instance.capacity_of_vehicles:
                    self.feasible = False
                    self.total_distance = INT_MAX
                    self.num_vehicles = INT_MAX
                    return

    def __deepcopy__(self, memodict: Dict=None) -> "FIGASolution":
        return FIGASolution(_id=self.id, vehicles=[copy.deepcopy(v) for v in self.vehicles], feasible=self.feasible, total_distance=self.total_distance, num_vehicles=self.num_vehicles)
