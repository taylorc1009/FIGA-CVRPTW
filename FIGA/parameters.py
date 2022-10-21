from typing import Final

POPULATION_SIZE: Final[int]=10
TERMINATION_CONDITION_ITERATIONS: Final[int]=350
TERMINATION_CONDITION_SECONDS: Final[int]=600 # 10 minutes
TERMINATION_CONDITION_TYPE: Final[str]="seconds" # can also be set to "iterations", thus using the TERMINATION_CONDITION_ITERATIONS parameter instead of TERMINATION_CONDITION_SECONDS
TEMPERATURE_MAX: Final[float]=100.0
TEMPERATURE_MIN: Final[float]=80.0
TEMPERATURE_STOP: Final[float]=20.0
NUM_PROGRESS_OUTPUTS: Final[int]=10 # number of times you wish to print the population/non-dominated set's state during runtime

CROSSOVER_PROBABILITY: Final[int]=80
MUTATION_PROBABILITY: Final[int]=50
TOURNAMENT_PROBABILITY_SELECT_BEST: Final[int]=20

ES_CROSSOVER_MAX_VEHICLES: Final[int]=3
SBRC_CROSSOVER_MAX_VEHICLES: Final[int]=2
FBR_CROSSOVER_MAX_VEHICLES: Final[int]=3
MAX_SIMULTANEOUS_MUTATIONS: Final[int]=3

MUTATION_LONGEST_WAIT_PROBABILITY: Final[int]=10 # used by TWBS
MUTATION_REVERSE_SWAP_PROBABILITY: Final[int]=33 # used by TWBS
MUTATION_LONGEST_ROUTE_PROBABILITY: Final[int]=10 # used by DBT, DBS, and LDHR
MUTATION_SWAP_PROBABILITY: Final[int]=50 # used by TWBS, DBT, DBS, and LDHR
MUTATION_FEASIBLE_SWAP_PROBABILITY: Final[int]=50 # used by FBS
MUTATION_MAX_FEASIBLE_SWAPS: Final[int]=5 # used by FBS
MUTATION_ELIMINATE_SHORTEST_PROBABILITY: Final[int]=10 # used by VE
MUTATION_SHORT_ROUTE_POOL_SIZE: Final[int]=3 # used by VE
# MUTATION_THREATENED_WINDOW_PROBABILITY: Final[int]=20 # used by ATBR
MUTATION_MAX_SLICE_LENGTH: Final[int]=5 # used by PBS