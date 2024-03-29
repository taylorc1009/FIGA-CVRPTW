from typing import Callable, List, Union

from common import check_are_identical
from constants import INT_MAX
from destination import Destination
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.ombukiSolution import OmbukiSolution
from problemInstance import ProblemInstance
from vehicle import Vehicle


def is_nondominated(parent: MMOEASASolution, child: MMOEASASolution) -> bool:
    return (child.total_distance < parent.total_distance and child.cargo_unbalance <= parent.cargo_unbalance) or (child.total_distance <= parent.total_distance and child.cargo_unbalance < parent.cargo_unbalance)

def ombuki_is_nondominated(old_solution: OmbukiSolution, new_solution: OmbukiSolution) -> bool:
    return (new_solution.total_distance < old_solution.total_distance and new_solution.num_vehicles <= old_solution.num_vehicles) or (new_solution.total_distance <= old_solution.total_distance and new_solution.num_vehicles < old_solution.num_vehicles)

def check_nondominated_set_acceptance(nondominated_set: List[Union[MMOEASASolution, OmbukiSolution]], subject_solution: Union[MMOEASASolution, OmbukiSolution], nondominated_check: Callable[[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]], bool]) -> bool:
    if not subject_solution.feasible:
        return# False

    nondominated_set.append(subject_solution)
    solutions_to_remove = set()

    # check commentary of "check_nondominated_set_acceptance" in "../FIGA/figa.py"
    for s, solution in enumerate(nondominated_set[:-1]):
        if s not in solutions_to_remove:
            for s_aux, solution_auxiliary in enumerate(nondominated_set[s + 1:], s + 1):
                if s_aux not in solutions_to_remove:
                    if nondominated_check(solution, solution_auxiliary):
                        solutions_to_remove.add(s)
                        break
                    elif nondominated_check(solution_auxiliary, solution) \
                            or check_are_identical(solution, solution_auxiliary):
                        solutions_to_remove.add(s_aux)

    if solutions_to_remove:
        i = 0
        for s in range(len(nondominated_set)):
            if s not in solutions_to_remove:
                nondominated_set[i] = nondominated_set[s]
                i += 1
        if i != len(nondominated_set):
            del nondominated_set[i:]

    # return subject_solution in nondominated_set

def insert_unvisited_node(solution: Union[MMOEASASolution, OmbukiSolution], instance: ProblemInstance, node: int) -> Union[MMOEASASolution, OmbukiSolution]:
    inserted = False
    vehicle = 0
    infeasible_vehicle, lowest_delay = -1, float(INT_MAX)

    while vehicle < len(solution.vehicles) and not inserted:
        if solution.vehicles[vehicle].current_capacity + instance.nodes[node].demand < instance.capacity_of_vehicles:
            position = solution.vehicles[vehicle].get_num_of_customers_visited() + 1
            solution.vehicles[vehicle].destinations.insert(position, Destination(instance.nodes[node]))
            solution.vehicles[vehicle].calculate_destination_time_window(instance, position - 1, position)

            if solution.vehicles[vehicle].destinations[position].arrival_time <= instance.nodes[node].due_date:
                inserted = True
                break
            elif not len(solution.vehicles) < instance.amount_of_vehicles and solution.vehicles[vehicle].destinations[position].wait_time < lowest_delay:
                infeasible_vehicle = vehicle # keeps track of the best vehicle to insert the infeasible destination to the end of
                lowest_delay = solution.vehicles[vehicle].destinations[position].wait_time # the best position will cause the lowest wait time required to service the infeasible vehicle

            solution.vehicles[vehicle].destinations.pop(position)
        if not inserted:
            vehicle += 1

    if not inserted: # in this case, the unvisited node doesn't fit into any of the existing routes, so it needs a new vehicle
        if len(solution.vehicles) < instance.amount_of_vehicles:
            new_vehicle = Vehicle.create_route(instance, node=instance.nodes[node])
            solution.vehicles.append(new_vehicle)
        else: # only use the best infeasible insertion point if the solution cannot allocate a new vehicle
            solution.vehicles[infeasible_vehicle].destinations.insert(len(solution.vehicles[infeasible_vehicle].destinations) - 1, Destination(node=instance.nodes[node]))
            vehicle = infeasible_vehicle

        # the calculations needed so that another invocation of "insert_unvisited_nodes()" will have the correct time windows when determining where to insert the next unvisited node
        solution.vehicles[vehicle].calculate_destinations_time_windows(instance)
        solution.vehicles[vehicle].calculate_vehicle_load()
        solution.vehicles[vehicle].calculate_length_of_route(instance)
    else:
        num_customers = solution.vehicles[vehicle].get_num_of_customers_visited()
        solution.vehicles[vehicle].calculate_destination_time_window(instance, num_customers, num_customers + 1)
        solution.vehicles[vehicle].current_capacity += instance.nodes[node].demand
        solution.vehicles[vehicle].route_distance += instance.get_distance(node, 0)

    return solution
