from typing import Final

POPULATION_SIZE: Final[int]=30
TERMINATION_CONDITION_ITERATIONS: Final[int]=350
TERMINATION_CONDITION_SECONDS: Final[int]=600 # 10 minutes
TERMINATION_CONDITION_TYPE: Final[str]="seconds" # can also be set to "iterations", thus using the TERMINATION_CONDITION_ITERATIONS parameter instead of TERMINATION_CONDITION_SECONDS
NUM_PROGRESS_OUTPUTS: Final[int]=10 # number of times you wish to print the population/non-dominated set's state during runtime
CROSSOVER_PROBABILITY: Final[int]=80 # 80%
MUTATION_PROBABILITY: Final[int]=50 # 50%
TOURNAMENT_PROBABILITY_SELECT_BEST: Final[int]=80 # 80%
MUTATION_LONGEST_WAIT_PROBABILITY: Final[int]=66 # 66%
MUTATION_LONGEST_ROUTE_PROBABILITY: Final[int]=66 # 66%
