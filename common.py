import time
from typing import Final, Set
from numpy import random, floor

INT_MAX: Final[int]=2147483647

def rand(start: int, end: int, exclude_values: Set[int]=None) -> int:
    # '+ 1' to make the random number generator inclusive of the "end" value
    offset = 1 if end < INT_MAX else 0
    random_val = random.randint(start, end + offset)
    while exclude_values is not None and random_val in exclude_values:
        random_val = random.randint(start, end + offset)
    return random_val

def check_seconds_termination_condition(start: float, termination_condition: int, nondominated_set_length: int):
    time_taken = time.time() - start
    # it's slightly difficult to output only one measurement of the time taken
    # the only way would be to create a list of times that a measurement should be outputted at and determine whether an output has been made for that time
    # but that would be more bother than it's worth
    if not floor(time_taken % (termination_condition / 10)):
        print(f"time_taken={round(time_taken, 1)}s, {nondominated_set_length=}")
    return not time_taken < termination_condition

def check_iterations_termination_condition(iterations: int, termination_condition: int, nondominated_set_length: int):
    if not iterations % (termination_condition / 10):
        print(f"{iterations=}, {nondominated_set_length=}")
    return not iterations < termination_condition
