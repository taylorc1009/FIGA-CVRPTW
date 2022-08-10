from time import process_time
from typing import Deque, List, Set
from numpy import random
from constants import INT_MAX
from solution import Solution

def rand(start: int, end: int, exclude_values: Set[int]=None) -> int:
    # '+ 1' to make the random number generator inclusive of the "end" value
    offset = 1 if end < INT_MAX else 0
    random_val = random.randint(start, end + offset)
    while exclude_values is not None and random_val in exclude_values:
        random_val = random.randint(start, end + offset)
    return random_val

def check_are_identical(solution_one: Solution, solution_two: Solution) -> bool:
    return [d.node.number for v in sorted(solution_one.vehicles, key=lambda v: v.destinations[1].node.number) for d in v.get_customers_visited()] \
        == [d.node.number for v in sorted(solution_two.vehicles, key=lambda v: v.destinations[1].node.number) for d in v.get_customers_visited()]

def num_unique_solutions(population: List[Solution]) -> int:
    unique_solutions = set()
    for s, solution in enumerate(population[:-1]): # len - 1 because in the next loop, s + 1 will do the comparison of the last non-dominated solution; we never need s and s_aux to equal the same value as there's no point comparing identical solutions
        for s_aux, solution_auxiliary in enumerate(population[s + 1:], s + 1):
            if check_are_identical(solution, solution_auxiliary):
                break
            elif s_aux == len(population) - 1:
                unique_solutions.add(s)
                if s == len(population) - 2:
                    unique_solutions.add(s_aux)
    return len(unique_solutions)

def check_seconds_termination_condition(start: float, termination_condition: int, nondominated_set_length: int, population: List[Solution], progress_indication_steps: Deque[float]) -> bool:
    time_taken = process_time() - start
    if progress_indication_steps and time_taken >= progress_indication_steps[0]:
        progress_indication_steps.popleft()
        print(f"time_taken={round(time_taken, 1)}s, {nondominated_set_length=}, {num_unique_solutions(population)=}")
    return not time_taken < termination_condition

def check_iterations_termination_condition(iterations: int, termination_condition: int, nondominated_set_length: int, population: List[Solution], progress_indication_steps: Deque[float]) -> bool:
    if progress_indication_steps and iterations >= progress_indication_steps[0]:
        progress_indication_steps.popleft()
        print(f"{iterations=}, {nondominated_set_length=}, {num_unique_solutions(population)=}")
    return not iterations < termination_condition
