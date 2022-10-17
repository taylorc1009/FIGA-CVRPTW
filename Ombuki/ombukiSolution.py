import copy
from typing import List, Dict
from constants import INT_MAX
from MMOEASA.constants import INFINITY
from vehicle import Vehicle
from problemInstance import ProblemInstance
from solution import Solution

class OmbukiSolution(Solution):
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, default_temperature: float=0.0, temperature: float=0.0, cooling_rate: float=0.0, total_distance: float=0.0, num_vehicles: int=0, rank: int=INT_MAX) -> None:
        super(OmbukiSolution, self).__init__(_id=_id, vehicles=vehicles, feasible=feasible, default_temperature=default_temperature, temperature=temperature, cooling_rate=cooling_rate, total_distance=total_distance, rank=rank) # initialise every solution's temperature (in Ombuki) to 100 and don't modify it later so that Simulated Annealing isn't used when MMOEASA's acceptance criterion, "MO_Metropolis", is being utilised
        self.num_vehicles: int=int(num_vehicles) # the reason this objective is a variable instead of just using "len(vehicles)" is because if the solution is invalid, it needs to be set to a very high number

    def __str__(self) -> str:
        return f"total_distance={self.total_distance}, num_vehicles={self.num_vehicles}, {self.fitness}, {len(self.vehicles)=}, {[f'{i}. {str(v)}' for i, v in enumerate(self.vehicles)]}"

    def __nullify(self) -> None:
        self.feasible = False
        self.total_distance = float(INFINITY)
        self.num_vehicles = INFINITY

    def objective_function(self, instance: ProblemInstance) -> None:
        if len(self.vehicles) > instance.amount_of_vehicles:
            self.__nullify()
            return

        self.total_distance = 0.0
        self.num_vehicles = len(self.vehicles)
        self.feasible = True # set the solution as feasible temporarily

        for vehicle in self.vehicles:
            if vehicle.current_capacity > instance.capacity_of_vehicles:
                self.__nullify()
                return

            self.total_distance += vehicle.route_distance
            for destination in vehicle.get_customers_visited():
                if destination.arrival_time > instance.nodes[destination.node.number].due_date:
                    self.__nullify()
                    return

    def __deepcopy__(self, memodict: Dict=None) -> "OmbukiSolution":
        return OmbukiSolution(_id=self.id, vehicles=[copy.deepcopy(v) for v in self.vehicles], feasible=self.feasible, default_temperature=self.default_temperature, temperature=self.temperature, cooling_rate=self.cooling_rate, total_distance=self.total_distance, num_vehicles=self.num_vehicles, rank=self.rank)

    def is_valid(self, filename: str) -> "OmbukiSolution":
        ...