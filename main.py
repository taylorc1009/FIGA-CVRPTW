import os
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import deque
from typing import List, Union, Tuple, Dict
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.ombukiSolution import OmbukiSolution
from FIGA.figaSolution import FIGASolution
from problemInstance import ProblemInstance
from data import open_problem_instance, write_solution_for_graph
from MMOEASA.mmoeasa import MMOEASA
from Ombuki.ombuki import Ombuki
from evaluation import calculate_area
from FIGA.figa import FIGA

def execute_MMOEASA(problem_instance: ProblemInstance) -> Tuple[List[Union[MMOEASASolution, OmbukiSolution]], Dict[str, int]]:
    from MMOEASA.parameters import POPULATION_SIZE, MULTI_STARTS, TERMINATION_CONDITION_ITERATIONS, TERMINATION_CONDITION_SECONDS, TERMINATION_CONDITION_TYPE, NUM_PROGRESS_OUTPUTS, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, TEMPERATURE_MAX, TEMPERATURE_MIN, TEMPERATURE_STOP
    return MMOEASA(problem_instance, POPULATION_SIZE, MULTI_STARTS, TERMINATION_CONDITION_SECONDS if TERMINATION_CONDITION_TYPE == "seconds" else TERMINATION_CONDITION_ITERATIONS, TERMINATION_CONDITION_TYPE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, TEMPERATURE_MAX, TEMPERATURE_MIN, TEMPERATURE_STOP, deque([(TERMINATION_CONDITION_SECONDS / 10) * step if TERMINATION_CONDITION_TYPE == "seconds" else (TERMINATION_CONDITION_ITERATIONS / 10) * step for step in range(NUM_PROGRESS_OUTPUTS)]))

def execute_Ombuki(problem_instance: ProblemInstance, use_original: bool) -> Tuple[List[Union[OmbukiSolution, MMOEASASolution]], Dict[str, int]]:
    from Ombuki.parameters import POPULATION_SIZE, TERMINATION_CONDITION_ITERATIONS, TERMINATION_CONDITION_SECONDS, TERMINATION_CONDITION_TYPE, NUM_PROGRESS_OUTPUTS, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY
    return Ombuki(problem_instance, POPULATION_SIZE, TERMINATION_CONDITION_SECONDS if TERMINATION_CONDITION_TYPE == "seconds" else TERMINATION_CONDITION_ITERATIONS, TERMINATION_CONDITION_TYPE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, use_original, deque([(TERMINATION_CONDITION_SECONDS / 10) * step if TERMINATION_CONDITION_TYPE == "seconds" else (TERMINATION_CONDITION_ITERATIONS / 10) * step for step in range(NUM_PROGRESS_OUTPUTS)]))

def execute_FIGA(problem_instance: ProblemInstance) -> Tuple[List[FIGASolution], Dict[str, int]]:
    from FIGA.parameters import POPULATION_SIZE, TERMINATION_CONDITION_ITERATIONS, TERMINATION_CONDITION_SECONDS, TERMINATION_CONDITION_TYPE, TEMPERATURE_MAX, TEMPERATURE_MIN, TEMPERATURE_STOP, NUM_PROGRESS_OUTPUTS, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY
    return FIGA(problem_instance, POPULATION_SIZE, TERMINATION_CONDITION_SECONDS if TERMINATION_CONDITION_TYPE == "seconds" else TERMINATION_CONDITION_ITERATIONS, TERMINATION_CONDITION_TYPE, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, TEMPERATURE_MAX, TEMPERATURE_MIN, TEMPERATURE_STOP, deque([(TERMINATION_CONDITION_SECONDS / 10) * step if TERMINATION_CONDITION_TYPE == "seconds" else (TERMINATION_CONDITION_ITERATIONS / 10) * step for step in range(NUM_PROGRESS_OUTPUTS)]))

if __name__ == '__main__':
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=
            f"Example commands are:{os.linesep}"
            f" - \"main.py MMOEASA solomon_100/C101.txt -ac MMOEASA\"{os.linesep}"
            f" - \"main.py FIGA solomon_100/RC102.txt\"{os.linesep}"
            f" - \"main.py Ombuki solomon_100/R201.txt -ac MMOEASA\""
    )

    parser.add_argument("algorithm",
        type=str,
        choices=["FIGA", "MMOEASA", "Ombuki", "Ombuki-Original"],
        help=f"The algorithms available are:{os.linesep}"
            f" - MMOEASA,{os.linesep}"
            f" - Ombuki,{os.linesep}"
            f" - Ombuki-Original - contains features that seem to be anomalous from the original research paper,{os.linesep}"
            f" - FIGA.{os.linesep}{os.linesep}"
    )

    parser.add_argument("problem_instance",
        type=str,
        help=f"There's multiple types of problems in Solomon's instances. Here's what they are:{os.linesep}"
            f" - Number of customers:{os.linesep}"
            f"   - 100 - 100 customers,{os.linesep}"
            f"   - Archived (and, therefore, out of order):{os.linesep}"
            f"     - 25 - 25 customers,{os.linesep}"
            f"     - 50 - 50 customers.{os.linesep}"
            f" - Customers' location:{os.linesep}"
            f"   - C - clustered customers,{os.linesep}"
            f"   - R - uniformly distributed customers,{os.linesep}"
            f"   - RC - a mix of R and C.{os.linesep}"
            f" - Width of customers' time windows:{os.linesep}"
            f"   - 1 - destinations with narrow time windows,{os.linesep}"
            f"   - 2 - destinations with wide time windows.{os.linesep}{os.linesep}"
            f"To execute a problem set, please enter a problem's filename. The details required, and the argument format, are:{os.linesep}"
            f" - solomon_[ number of customers ]/[ customers' location ][ width of time windows ]XX.txt,{os.linesep}"
            f" - Where XX is the instance number; see the folder \"solomon_[ number of customers ]\" for available instances.{os.linesep}{os.linesep}"
    )

    parser.add_argument("-ac", "--acceptance_criterion",
        type=str,
        choices=["MMOEASA", "Ombuki"],
        help=f"The acceptance criteria available are:{os.linesep}"
            f" - MMOEASA,{os.linesep}"
            f" - Ombuki.{os.linesep}{os.linesep}"
            f"FIGA should not be used as and does not accept an alternative acceptance criterion. It uses Ombuki's criterion by default.{os.linesep}{os.linesep}"
    )

    parser.add_argument("-r", "--runs",
        type=int,
        default=1,
        dest="runs",
        help="Number of runs you wish your given configuration to complete. Giving a value > 1 will cause your selected algorithm to run multiple times on your selected problem instance and acceptance criteria."
    )

    args = parser.parse_args()

    if args.algorithm == "FIGA":
        assert args.acceptance_criterion is None
        args.acceptance_criterion = "Ombuki"
    else:
        assert args.acceptance_criterion is not None

    problem_instance = open_problem_instance(args.algorithm, args.problem_instance, args.acceptance_criterion)
    results = []

    for run in range(args.runs):
        if args.runs > 1:
            print(f"Started run {run + 1} of {args.runs}")

        nondominated_set, statistics = None, None
        match args.algorithm:
            case "MMOEASA":
                nondominated_set, statistics = execute_MMOEASA(problem_instance)
            case "Ombuki-Original":
                nondominated_set, statistics = execute_Ombuki(problem_instance, True)
            case "Ombuki":
                nondominated_set, statistics = execute_Ombuki(problem_instance, False)
            case "FIGA":
                nondominated_set, statistics = execute_FIGA(problem_instance)

        # uncomment this code if you'd like a solution to be written to a CSV
        # solutions can be plotted on a scatter graph in Excel as the x and y coordinates of each vehicle's destinations are outputted and in the order that they are serviced
        # if nondominated_set:
        #    write_solution_for_graph(nondominated_set[0])

        for solution in nondominated_set:
            print(f"{os.linesep + str(solution)}")
        """print(f"{os.linesep}Algorithm \"{sys.argv[1]}'s\" statistics:")
        for statistic, value in statistics.items():
            print(f" - {statistic}: {str(value)}")
        print(f"{os.linesep + str(problem_instance)}")"""
        result = statistics["initialiser_execution_time"], statistics["feasible_initialisations"], calculate_area(problem_instance, nondominated_set, args.acceptance_criterion)
        results.append(result)

    if args.runs > 1:
        print(f"All runs completed. Results:")
        for result in results:
            print(f" - {result[0]}, {result[1]}, {result[2]}%")