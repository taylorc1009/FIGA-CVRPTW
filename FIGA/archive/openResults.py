import copy
import os
import re
import sys
from abc import ABC, abstractclassmethod, abstractmethod
from argparse import ArgumentParser, RawTextHelpFormatter
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple, Union

INFINITY = 7654321
INT_MAX = 2147483647

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

class ProblemInstance:
    distances: List[float]=None
    Hypervolume_total_distance: float=0.0
    Hypervolume_distance_unbalance: float=0.0 # currently, the distance unbalance objective is unused in the objective function, but this may change
    Hypervolume_cargo_unbalance: float=0.0
    Hypervolume_num_vehicles: float=0.0

    def __init__(self, name: str, amount_of_vehicles: int, capacity_of_vehicles: int, nodes: Dict[int, Node]=None, acceptance_criterion: str="") -> None:
        self.name: str=name
        self.amount_of_vehicles: int=int(amount_of_vehicles)
        self.capacity_of_vehicles: int=int(capacity_of_vehicles)
        self.nodes: Dict[int, Node]=nodes
        self.acceptance_criterion: str=str(acceptance_criterion)

    def __str__(self) -> str:
        return f"ProblemInstance(name={self.name}, amount_of_vehicles={self.amount_of_vehicles}, capacity_of_vehicles={self.capacity_of_vehicles}, {len(self.nodes)=}"#, {[(key, str(value)) for key, value in self.nodes.items()]})"
    
    def calculate_distances(self) -> None:
        n = len(self.nodes)
        self.distances: List[float]=[-1.0 for _ in range(0, n ** 2)]

        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    self.distances[n * i + j] = self.nodes[i].get_distance(self.nodes[j].x, self.nodes[j].y)

    def get_distance(self, from_node: int, to_node: int) -> float:
        return self.distances[len(self.nodes) * from_node + to_node]

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

class Solution(ABC):
    def __init__(self, _id: int=None, vehicles: List[Vehicle]=None, feasible: bool=True, total_distance: float=0.0, rank: int=INT_MAX, default_temperature: float=0.0, temperature: float=0.0, cooling_rate: float=0.0) -> None:
        self.id: int=int(_id)
        self.vehicles: List[Vehicle]=vehicles if vehicles else []
        self.feasible: bool=feasible
        self.total_distance: float=float(total_distance)

        # Simulated Annealing parameters for MMOEASA: exists in the parent class for when MMOEASA is solving Ombuki's objective function
        self.default_temperature: float=float(default_temperature)
        self.temperature: float=float(temperature)
        self.cooling_rate: float=float(cooling_rate)

        # ranking used in Ombuki's Algorithm; exists in the parent class for when Ombuki's Algorithm is solving MMOEASA's objective function
        self.rank: int=int(rank)

    def calculate_routes_time_windows(self, instance: ProblemInstance) -> None:
        for vehicle in self.vehicles:
            vehicle.calculate_destinations_time_windows(instance)

    def calculate_vehicles_loads(self) -> None:
        for vehicle in self.vehicles:
            vehicle.calculate_vehicle_load()

    def calculate_length_of_routes(self, instance: ProblemInstance) -> None:
        for vehicle in self.vehicles:
            vehicle.calculate_length_of_route(instance)

    def check_format_is_correct(self, instance: ProblemInstance) -> None:
        node_nums = set(range(1, 101))
        # error checks to ensure that every route is of the valid format
        if len(self.vehicles) > instance.amount_of_vehicles:
            raise ValueError(f"Too many vehicles: {len(self.vehicles)}")
        elif sum(v.get_num_of_customers_visited() for v in self.vehicles) != len(instance.nodes) - 1: # check if the solution contains the correct amount of destinations, in that it visits all of them (this will also find depot returns mid route)
            raise ValueError(f"Mismatched amount of destinations: {sum(v.get_num_of_customers_visited() for v in self.vehicles)}")
        elif [v for v in self.vehicles if len(v.destinations) < 3]: # checks if all the routes have at least 3 destinations; routes should always at least depart from and return to the depot, while visiting a customer inbetween
            raise ValueError(f"Number of destinations is not at least 3 in the following vehicle(s): {str([i for i, v in enumerate(self.vehicles) if len(v.destinations) < 3])}")
        elif [v for v in self.vehicles if v.destinations[0].node.number or v.destinations[-1].node.number]: # checks that every route starts and ends at the depot
            raise ValueError(f"Indexes 0 and n - 1 are not depot nodes in the following vehicle(s): {str([i for i, v in enumerate(self.vehicles) if v.destinations[0].node.number or v.destinations[-1].node.number])}")
        elif node_nums.difference(set(d.node.number for v in self.vehicles for d in v.get_customers_visited())): # checks if all nodes have been visited; ".remove" will also find both: duplicate nodes as it will throw an exception when it tries to remove an already-removed node, and depot nodes in the middle of a route as the set starts at 1, so if it tries to remove the depot (node 0) it won't exist and throw an exception
            raise ValueError("Not all nodes are visited")

    @abstractmethod
    def objective_function(self, instance: ProblemInstance) -> None:
        ...

    @abstractclassmethod
    def is_valid(cls, filename: str) -> "Solution":
        ...

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
        relative_path =  os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else f"{str(Path(__file__).parent.resolve())}\\{filename}"
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

def open_problem_instance(algorithm: str, filename: str, acceptance_criterion: str) -> ProblemInstance:
    try:
        with open(filename, 'r') as file:
            problem_instance = None
            problem_name = file.readline().strip() # because the problem name is the first line in the text files, this line quickly adds it to a variable (so we can add it to a "ProblemInstance" object later"
            next(file) # skips the first line (containing the problem name), preventing it from being iterated over

            for line in file:
                if line is not None and not re.search('[a-zA-Z]', line): # checks if the current line contains any characters; we don't want to work with any of the alphabetical characters in the text files, only the numbers
                    cur_line = line.split()
                    if cur_line: # prevents any work being done with empty lines (lines that contained only a new line; '\n')
                        if len(cur_line) == 2: # if the current line only contains two numbers then it's the line that holds only the amount of vehicles and vehicles' capacity, so use them to make a "ProblemInstance"
                            problem_instance = ProblemInstance(problem_name, *cur_line, nodes=dict(), acceptance_criterion=acceptance_criterion)
                        else: # if the current line doesn't contain only two values, it will, instead, always contain seven and lines with seven values represent destinations
                            node = Node(*cur_line)
                            problem_instance.nodes[int(node.number)] = node
        problem_instance.calculate_distances()
        return problem_instance
    except FileNotFoundError as e:
        exc = FileNotFoundError(f"Couldn't open file \"{filename}\"\nCause: {e}")
        raise exc from None

def open_results(filename: str) -> Tuple[List["FIGASolution"], List[float], List["FIGASolution"], float]:
    relative_path =  os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else f"{str(Path(__file__).parent.resolve())}\\{filename}"
    solutions, areas, final_nondominated_set, final_nondominated_set_area = [], [], [], 0.0
    subject_list = solutions

    try:
        with open(relative_path, 'r') as file:
            problem_path = file.readline().strip() # because the problem name is the first line in the text files, this line quickly adds it to a variable (so we can add it to a "ProblemInstance" object later"
            instance = open_problem_instance("FIGA", problem_path, "Ombuki")

            line = file.readline().strip()
            while line:
                if line[0] in {'-', '/', '#'}:
                    if line[0] == '/': # this is the beginning of the final non-dominated set
                        subject_list = final_nondominated_set
                        final_nondominated_set_area = line[1:] # the Hypervolume is stored as a percentage (string with '%' symbol)
                    elif line[0] == '-': # this is the beginning of a new run
                        areas.append(line[1:]) # the Hypervolume is stored as a percentage (string with '%' symbol)
                    line = file.readline().strip().split(',') # lines after lines beginning with '#' and '-' should always be the total distance and number of vehicles of the next solution
                    subject_list.append(FIGASolution(_id=len(subject_list), total_distance=float(line[0]), num_vehicles=int(line[1])))
                else:
                    subject_list[-1].vehicles.append(Vehicle.create_route(instance, [instance.nodes[int(n)] for n in line.split(',')]))
                line = file.readline().strip()
    except FileNotFoundError as e:
        exc = FileNotFoundError(f"Couldn't open file \"{filename}\"\nCause: {e}")
        raise exc from None

    return solutions, areas, final_nondominated_set, final_nondominated_set_area

if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Loads results recorded from runs of EAs"
    )

    parser.add_argument("CSV",
        type=str,
        help=f"The name CSV file you wish to load; one which contains results of the EAs"
    )

    args = parser.parse_args()
    print(open_results(args.CSV)) # the print simply exists to show that the results were loaded