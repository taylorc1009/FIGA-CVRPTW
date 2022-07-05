from typing import Final

POPULATION_SIZE: Final[int]=20
MULTI_STARTS: Final[int]=10
TERMINATION_CONDITION_ITERATIONS: Final[int]=200 # if you use "iterations" as the termination condition, then the number of iterations that will be performed will be equal to TERMINATION_CONDITION_ITERATIONS * MULTI_STARTS
TERMINATION_CONDITION_SECONDS: Final[int]=600
TERMINATION_CONDITION_TYPE: Final[str]="seconds" # can also be set to "iterations", thus using the TERMINATION_CONDITION_ITERATIONS parameter instead of TERMINATION_CONDITION_SECONDS
CROSSOVER_PROBABILITY: Final[int]=25 # 25%
MUTATION_PROBABILITY: Final[int]=25 # 25%
TEMPERATURE_MAX: Final[float]=100.0
TEMPERATURE_MIN: Final[float]=50.0
TEMPERATURE_STOP: Final[float]=30.0
