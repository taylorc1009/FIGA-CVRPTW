from time import process_time
from typing import Deque, Final, List, Set
from numpy import random

INT_MAX: Final[int]=2147483647

def rand(start: int, end: int, exclude_values: Set[int]=None) -> int:
    # '+ 1' to make the random number generator inclusive of the "end" value
    offset = 1 if end < INT_MAX else 0
    random_val = random.randint(start, end + offset)
    while exclude_values is not None and random_val in exclude_values:
        random_val = random.randint(start, end + offset)
    return random_val

def check_seconds_termination_condition(start: float, termination_condition: int, nondominated_set_length: int, population: List, progress_indication_steps: Deque[float]) -> bool:
    time_taken = process_time() - start
    if progress_indication_steps and time_taken >= progress_indication_steps[0]:
        progress_indication_steps.popleft()
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

def check_iterations_termination_condition(iterations: int, termination_condition: int, nondominated_set_length: int, population: List, progress_indication_steps: Deque[float]) -> bool:
    if progress_indication_steps and iterations >= progress_indication_steps[0]:
        progress_indication_steps.popleft()
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
