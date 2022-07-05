from typing import Final

POPULATION_SIZE: Final[int]=30
TERMINATION_CONDITION_ITERATIONS: Final[int]=350
TERMINATION_CONDITION_SECONDS: Final[int]=600 # 10 minutes
TERMINATION_CONDITION_TYPE: Final[str]="seconds" # can also be set to "iterations", thus using the TERMINATION_CONDITION_ITERATIONS parameter instead of TERMINATION_CONDITION_SECONDS
CROSSOVER_PROBABILITY: Final[int]=80 # 80%
MUTATION_PROBABILITY: Final[int]=50 # 50%
