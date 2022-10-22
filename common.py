from multiprocessing import Process
from random import choice
from numpy.random import randint
from time import process_time
from typing import Deque, List, Set, Tuple
from constants import INT_MAX
from solution import Solution
from difflib import SequenceMatcher

def rand(start: int, end: int, exclude_values: Set[int]=None) -> int:
    # '+ 1' to make the random number generator inclusive of the "end" value
    if start == end:
        return start
    if not exclude_values:
        return randint(start, end)
    return choice(list(set(range(start, end + (1 if end < INT_MAX else 0))).difference(exclude_values if exclude_values else set())))

def check_are_identical(solution_one: Solution, solution_two: Solution) -> bool: # this is kept separate from "evaluate_similarity" below as not every function needs to know solutions' similarity rating, just whether they're identical or not
    return [d.node.number for v in sorted(solution_one.vehicles, key=lambda v: v.destinations[1].node.number) for d in v.get_customers_visited()] \
        == [d.node.number for v in sorted(solution_two.vehicles, key=lambda v: v.destinations[1].node.number) for d in v.get_customers_visited()]

def evaluate_similarity(solution_one: Solution, solution_two: Solution) -> Tuple[bool, float]:
    s_one, s_two = [d.node.number for v in sorted(solution_one.vehicles, key=lambda v: v.destinations[1].node.number) for d in v.get_customers_visited()], [d.node.number for v in sorted(solution_two.vehicles, key=lambda v: v.destinations[1].node.number) for d in v.get_customers_visited()]
    if s_one == s_two:
        return True, 100.0
    return False, SequenceMatcher(None, s_one, s_two).ratio() * 100.0 #(sum(1 if n == s_two[i] else 0 for i, n in enumerate(s_one)) / len(s_one)) * 100

def evaluate_population(population: List[Solution]) -> Tuple[int, float]:
    similarities = []
    unique_solutions = set()
    for s, solution in enumerate(population[:-1]): # len - 1 because in the next loop, s + 1 will do the comparison of the last non-dominated solution; we never need s and s_aux to equal the same value as there's no point comparing identical solutions
        for s_aux, solution_auxiliary in enumerate(population[s + 1:], s + 1):
            identical, similarity = evaluate_similarity(solution, solution_auxiliary)
            similarities.append(similarity)
            if identical:
                break
            elif s_aux == len(population) - 1:
                unique_solutions.add(s)
                if s == len(population) - 2:
                    unique_solutions.add(s_aux)
    return len(unique_solutions), round(sum(similarities) / len(similarities), 3)

def output_status_seconds(nondominated_set_length: int, population: List[Solution], time_taken: float) -> None:
    unique_solutions, average_similarity = evaluate_population(population)
    population_str = ""
    for solution in population:
        bullet = "-" if solution.feasible else "!"
        population_str += f" {bullet} {solution.total_distance},{len(solution.vehicles)}\n"
    print(f"time_taken={round(time_taken, 1)}s, {nondominated_set_length=}, {unique_solutions=}, {average_similarity=}%, \n{population_str}")

def check_seconds_termination_condition(start: float, termination_condition: int, nondominated_set_length: int, population: List[Solution], progress_indication_steps: Deque[float]) -> bool:
    # time_taken = process_time() - start
    # if progress_indication_steps and time_taken >= progress_indication_steps[0]:
    #     progress_indication_steps.popleft()
    #     Process(target=output_status_seconds, args=(nondominated_set_length, population, time_taken)).start()
    # return not time_taken < termination_condition
    return not (process_time() - start) < termination_condition

def output_status_iterations(nondominated_set_length: int, population: List[Solution], iterations: int) -> None:
    unique_solutions, average_similarity = evaluate_population(population)
    print(f"{iterations=}, {nondominated_set_length=}, {unique_solutions=}, {average_similarity=}%")

def check_iterations_termination_condition(iterations: int, termination_condition: int, nondominated_set_length: int, population: List[Solution], progress_indication_steps: Deque[float]) -> bool:
    # if progress_indication_steps and iterations >= progress_indication_steps[0]:
    #     progress_indication_steps.popleft()
    #     Process(target=output_status_iterations, args=(nondominated_set_length, population, iterations)).start()
    return not iterations < termination_condition
