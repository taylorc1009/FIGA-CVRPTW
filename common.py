import time
from typing import Final, List, Set
from numpy import random, floor

INT_MAX: Final[int]=2147483647

def rand(start: int, end: int, exclude_values: Set[int]=None) -> int:
    # '+ 1' to make the random number generator inclusive of the "end" value
    offset = 1 if end < INT_MAX else 0
    random_val = random.randint(start, end + offset)
    while exclude_values is not None and random_val in exclude_values:
        random_val = random.randint(start, end + offset)
    return random_val

def check_seconds_termination_condition(start: float, termination_condition: int, nondominated_set_length: int, population: List) -> bool:
    time_taken = time.time() - start
    # it's slightly difficult to output only one measurement of the time taken
    # the only way would be to create a list of times that a measurement should be outputted at and determine whether an output has been made for that time
    # but that would be more bother than it's worth
    if not floor(time_taken % (termination_condition / 10)):
        unique_solutions: Set[str] = set()
        for solution in population:
            objectives = ""
            if hasattr(solution, "distance_unbalance") or hasattr(solution, "cargo_unbalance"):
                objectives = f"{solution.total_distance},{solution.distance_unbalance},{solution.cargo_unbalance}"
            else: # elif hasattr(solution, "num_vehicles"):
                objectives = f"{solution.total_distance},{solution.num_vehicles}"
            if objectives not in unique_solutions:
                unique_solutions.add(objectives)
        print(f"time_taken={round(time_taken, 1)}s, {nondominated_set_length=}, {len(unique_solutions)=}")
    return not time_taken < termination_condition

def check_iterations_termination_condition(iterations: int, termination_condition: int, nondominated_set_length: int, population: List) -> bool:
    if not iterations % (termination_condition / 10):
        unique_solutions: Set[str] = set()
        for solution in population:
            objectives = ""
            if hasattr(solution, "distance_unbalance") or hasattr(solution, "cargo_unbalance"):
                objectives = f"{solution.total_distance},{solution.distance_unbalance},{solution.cargo_unbalance}"
            else: # elif hasattr(solution, "num_vehicles"):
                objectives = f"{solution.total_distance},{solution.num_vehicles}"
            if objectives not in unique_solutions:
                unique_solutions.add(objectives)
        print(f"{iterations=}, {nondominated_set_length=}, {len(unique_solutions)=}")
    return not iterations < termination_condition
