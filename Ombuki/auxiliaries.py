from typing import List, Union, Callable
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.ombukiSolution import OmbukiSolution
from common import check_are_identical

def is_nondominated(old_solution: OmbukiSolution, new_solution: OmbukiSolution) -> bool:
    return (new_solution.total_distance < old_solution.total_distance and new_solution.num_vehicles <= old_solution.num_vehicles) or (new_solution.total_distance <= old_solution.total_distance and new_solution.num_vehicles < old_solution.num_vehicles)
    #return (new_solution.total_distance < old_solution.total_distance and new_solution.num_vehicles <= old_solution.num_vehicles) or (new_solution.total_distance <= old_solution.total_distance and new_solution.num_vehicles < old_solution.num_vehicles)

def mmoeasa_is_nondominated(parent: MMOEASASolution, child: MMOEASASolution) -> bool:
    return (child.total_distance < parent.total_distance and child.cargo_unbalance <= parent.cargo_unbalance) or (child.total_distance <= parent.total_distance and child.cargo_unbalance < parent.cargo_unbalance)

def get_nondominated_set(population: List[Union[OmbukiSolution, MMOEASASolution]], nondominated_check: Callable[[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]], bool]) -> List[Union[OmbukiSolution, MMOEASASolution]]:
    nondominated_set = set(range(len(population)))

    # check commentary of "check_nondominated_set_acceptance" in "../FIGA/figa.py"
    for s, solution in enumerate(population[:-1]):
        if s in nondominated_set:
            for s_aux, solution_auxiliary in enumerate(population[s + 1:], s + 1):
                if s_aux in nondominated_set:
                    if nondominated_check(solution, solution_auxiliary):
                        nondominated_set.remove(s)
                        break
                    elif nondominated_check(solution_auxiliary, solution):
                        nondominated_set.remove(s_aux)

    return [population[i] for i in nondominated_set]

def get_unique_set(population: List[Union[OmbukiSolution, MMOEASASolution]]) -> List[Union[OmbukiSolution, MMOEASASolution]]:
    unique_set = set(range(len(population)))

    # check commentary of "check_nondominated_set_acceptance" in "../FIGA/figa.py"
    for s, solution in enumerate(population[:-1]):
        if s in unique_set:
            for s_aux, solution_auxiliary in enumerate(population[s + 1:], s + 1):
                if s_aux in unique_set:
                    if check_are_identical(solution, solution_auxiliary):
                        unique_set.remove(s)
                        break

    return [population[i] for i in unique_set]