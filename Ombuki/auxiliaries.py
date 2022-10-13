from typing import Set, List, Union, Callable
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.ombukiSolution import OmbukiSolution

def is_nondominated(old_solution: OmbukiSolution, new_solution: OmbukiSolution) -> bool:
    return new_solution.fitness < old_solution.fitness
    #return (new_solution.total_distance < old_solution.total_distance and new_solution.num_vehicles <= old_solution.num_vehicles) or (new_solution.total_distance <= old_solution.total_distance and new_solution.num_vehicles < old_solution.num_vehicles)

def mmoeasa_is_nondominated(parent: MMOEASASolution, child: MMOEASASolution) -> bool:
    return (child.total_distance < parent.total_distance and child.cargo_unbalance <= parent.cargo_unbalance) or (child.total_distance <= parent.total_distance and child.cargo_unbalance < parent.cargo_unbalance)

def get_lowest_weighted_solution(population: List[Union[OmbukiSolution, MMOEASASolution]], nondominated_check: Callable[[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]], bool]) -> int:
    nondominated_solution = None

    # check commentary of "check_nondominated_set_acceptance" in "../FIGA/figa.py"
    for s, solution in enumerate(population[:-1]):
        for s_aux, solution_auxiliary in enumerate(population[s + 1:], s + 1):
            if nondominated_check(solution, solution_auxiliary):
                nondominated_solution = s
            elif nondominated_check(solution_auxiliary, solution):
                nondominated_solution = s_aux

    return nondominated_solution
