import copy
import os
import sys
from typing import List, Dict
from data import open_problem_instance
from vehicle import Vehicle
from problemInstance import ProblemInstance
from solution import Solution
from pathlib import Path
from MMOEASA.constants import INFINITY

class FIGASolution(Solution):
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, total_distance: float=0.0, num_vehicles: int=0, temperature: float=0.0, default_temperature: float=0.0, cooling_rate: float=0.0) -> None:
        super(FIGASolution, self).__init__(_id=_id, vehicles=vehicles, feasible=feasible, total_distance=total_distance, temperature=temperature, default_temperature=default_temperature, cooling_rate=cooling_rate)
        self.num_vehicles: int=int(num_vehicles) # the reason this objective is a variable instead of just using "len(vehicles)" is because if the solution is invalid, it needs to be set to a very high number

    def __str__(self) -> str:
        return f"total_distance={self.total_distance}, num_vehicles={self.num_vehicles}, {len(self.vehicles)=}, {[str(v) for v in sorted(self.vehicles, key=lambda v: v.destinations[1].node.number)]}"

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
                if destination.arrival_time > destination.node.due_date:
                    self.__nullify()
                    return

    def __deepcopy__(self, memodict: Dict=None) -> "FIGASolution":
        return FIGASolution(_id=self.id, vehicles=[copy.deepcopy(v) for v in self.vehicles], feasible=self.feasible, total_distance=self.total_distance, num_vehicles=self.num_vehicles, temperature=self.temperature, default_temperature=self.default_temperature, cooling_rate=self.cooling_rate)

    @classmethod
    def is_valid(cls, filename: str) -> "FIGASolution":
        relative_path = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else f"{str(Path(__file__).parent.resolve())}\\{filename}"
        solution = cls(_id=0)

        try:
            with open(relative_path, 'r') as file:
                problem_path = file.readline().strip() # because the problem name is the first line in the text files, this line quickly adds it to a variable (so we can add it to a "ProblemInstance" object later"
                instance = open_problem_instance("FIGA", problem_path, "Ombuki")
                for line in file:
                    cur_line = line.split()[0]
                    solution.vehicles.append(Vehicle.create_route(instance, [instance.nodes[int(n)] for n in cur_line.split(',')]))
        except FileNotFoundError as e:
            exc = FileNotFoundError(f"Couldn't open file \"{filename}\"\nCause: {e}")
            raise exc from None

        solution.calculate_length_of_routes(instance)
        solution.calculate_vehicles_loads()
        solution.calculate_routes_time_windows(instance)
        solution.objective_function(instance)

        return solution
