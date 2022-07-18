from typing import Final

POPULATION_SIZE: Final[int]=300
TERMINATION_CONDITION_ITERATIONS: Final[int]=350
TERMINATION_CONDITION_SECONDS: Final[int]=600
TERMINATION_CONDITION_TYPE: Final[str]="seconds" # can also be set to "iterations", thus using the TERMINATION_CONDITION_ITERATIONS parameter instead of TERMINATION_CONDITION_SECONDS
NUM_PROGRESS_OUTPUTS: Final[int]=10 # number of times you wish to print the population/non-dominated set's state during runtime
CROSSOVER_PROBABILITY: Final[int]=80 # 80%
MUTATION_PROBABILITY: Final[int]=10 # 10%
