import copy
from concurrent.futures import ProcessPoolExecutor
from random import choice, shuffle
from typing import Tuple, Union

from common import rand
from constants import INT_MAX
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.constants import MUTATION_REVERSAL_LENGTH
from Ombuki.ombukiSolution import OmbukiSolution
from problemInstance import ProblemInstance
from vehicle import Vehicle


def set_up_crossover_child(instance: ProblemInstance, crossover_solution: Union[OmbukiSolution, MMOEASASolution], parent_two_vehicle: Vehicle) -> Union[OmbukiSolution, MMOEASASolution]:
    # check commentary of "set_up_crossover_child" in "../FIGA/operators.py"
    nodes_to_remove = set([d.node.number for d in parent_two_vehicle.get_customers_visited()])
    i = 0
    while i < len(crossover_solution.vehicles) and nodes_to_remove:
        increment = True
        j = 1
        while j <= crossover_solution.vehicles[i].get_num_of_customers_visited() and nodes_to_remove:
            destination = crossover_solution.vehicles[i].destinations[j]
            if destination.node.number in nodes_to_remove:
                nodes_to_remove.remove(destination.node.number)
                crossover_solution.vehicles[i].current_capacity -= destination.node.demand
                if crossover_solution.vehicles[i].get_num_of_customers_visited() - 1 > 0:
                    del crossover_solution.vehicles[i].destinations[j]
                else:
                    increment = False
                    del crossover_solution.vehicles[i]
                    break # break, otherwise the while loop will start searching the next vehicle with "j" as the same value; without incrementing "i" and starting "j" at 0
            else:
                j += 1
        if increment:
            i += 1

    crossover_solution.calculate_routes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads()

    return crossover_solution

def original_crossover(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution], parent_vehicle: Vehicle) -> Union[OmbukiSolution, MMOEASASolution]:
    # check commentary of "crossover" in "../FIGA/operators.py"
    # the difference in this operator is that when no feasible insertion point is found and the amount of vehicles in the solution is at the limit, a new one is created anyway (which is bad)
    crossover_solution = set_up_crossover_child(instance, solution, parent_vehicle)

    randomized_destinations = list(range(1, len(parent_vehicle.destinations) - 1))
    shuffle(randomized_destinations)
    for d in randomized_destinations:
        parent_destination = parent_vehicle.destinations[d]
        best_vehicle, best_position = instance.amount_of_vehicles, 0
        shortest_distance = float(INT_MAX)
        found_feasible_location = False

        for i, vehicle in enumerate(crossover_solution.vehicles):
            if not vehicle.current_capacity + parent_destination.node.demand > instance.capacity_of_vehicles:
                for j in range(1, len(crossover_solution.vehicles[i].destinations)):
                    vehicle.destinations.insert(j, copy.deepcopy(parent_destination))

                    if vehicle.calculate_destinations_time_windows(instance, start_from=j):
                        vehicle.calculate_length_of_route(instance)
                        if vehicle.route_distance < shortest_distance:
                            best_vehicle, best_position, shortest_distance = i, j, vehicle.route_distance
                            found_feasible_location = True
                    
                    vehicle.destinations.pop(j)
                    vehicle.calculate_destinations_time_windows(instance, start_from=j)


        if not found_feasible_location:
            best_vehicle = len(crossover_solution.vehicles)
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination.node))
        else:
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].calculate_vehicle_load()
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    crossover_solution.objective_function(instance)
    return crossover_solution

def modified_crossover(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution], parent_vehicle: Vehicle) -> Union[OmbukiSolution, MMOEASASolution]:
    # check commentary of "crossover" in "../FIGA/operators.py"
    # the difference in this operator is that when no feasible insertion point is found and the amount of vehicles in the solution is at the limit, the destination to be inserted is appended to the end of the route where the route's last destination is nearest to the deatination to be inserted
    crossover_solution = set_up_crossover_child(instance, solution, parent_vehicle)

    randomized_destinations = list(range(1, len(parent_vehicle.destinations) - 1))
    shuffle(randomized_destinations)
    for d in randomized_destinations:
        parent_destination = parent_vehicle.destinations[d]
        best_vehicle, best_position = instance.amount_of_vehicles, 0
        shortest_distance = float(INT_MAX)
        found_feasible_location = False

        for i, vehicle in enumerate(crossover_solution.vehicles):
            if not vehicle.current_capacity + parent_destination.node.demand > instance.capacity_of_vehicles:
                for j in range(1, len(crossover_solution.vehicles[i].destinations)):
                    vehicle.destinations.insert(j, copy.deepcopy(parent_destination))

                    if vehicle.calculate_destinations_time_windows(instance, start_from=j):
                        vehicle.calculate_length_of_route(instance)
                        if vehicle.route_distance < shortest_distance:
                            best_vehicle, best_position, shortest_distance = i, j, vehicle.route_distance
                            found_feasible_location = True
                    
                    vehicle.destinations.pop(j)
                    vehicle.calculate_destinations_time_windows(instance, start_from=j)

        if not found_feasible_location:
            if len(crossover_solution.vehicles) < instance.amount_of_vehicles:
                best_vehicle = len(crossover_solution.vehicles)
                crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination.node))
            else:
                sorted_by_nearest = sorted(enumerate(crossover_solution.vehicles), key=lambda veh: instance.get_distance(veh[1].destinations[-2].node.number, parent_destination.node.number))
                for v, infeasible_vehicle in sorted_by_nearest:
                    if infeasible_vehicle.current_capacity + parent_destination.node.demand <= instance.capacity_of_vehicles:
                        infeasible_vehicle.destinations.insert(infeasible_vehicle.get_num_of_customers_visited() + 1, copy.deepcopy(parent_destination))
                        best_vehicle, best_position = v, infeasible_vehicle.get_num_of_customers_visited()
                        break
        else:
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].calculate_vehicle_load()
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    crossover_solution.objective_function(instance)
    return crossover_solution

def crossover(instance: ProblemInstance, parent_one: Union[OmbukiSolution, MMOEASASolution], parent_two: Union[OmbukiSolution, MMOEASASolution], use_original: bool) -> Tuple[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]]:
    # threads cannot return values, so they need to be given a mutable type that can be given the values we'd like to return; in this instance, a dict is used and the return values are assigned using the thread names
    # threading is used because Ombuki's crossover creates two child solutions
    """parent_one_vehicle, parent_two_vehicle = choice(parent_one.vehicles), choice(parent_two.vehicles)
    with ProcessPoolExecutor() as executor:
        crossover_one = executor.submit(original_crossover_thread if use_original else modified_crossover_thread, instance, parent_one, parent_two_vehicle)
        crossover_two = executor.submit(original_crossover_thread if use_original else modified_crossover_thread, instance, parent_two, parent_one_vehicle)
        child_one = crossover_one.result()
        child_two = crossover_two.result()"""

    operator = original_crossover if use_original else modified_crossover
    return operator(instance, parent_one, choice(parent_two.vehicles)), operator(instance, parent_two, choice(parent_one.vehicles))

"""def get_next_vehicles_destinations(solution: Union[OmbukiSolution, MMOEASASolution], vehicle: int, first_destination: int, remaining_destinations: int) -> List[Destination]:
    if not remaining_destinations: # if the amount of destinations left to acquire is equal to zero, then return an empty list
        return list()
    num_customers = solution.vehicles[vehicle].get_num_of_customers_visited()
    if num_customers < first_destination + remaining_destinations: # if the vehicle does not contain "remaining_destinations" amount of nodes, starting from "first_destination" position in the list, then we need to move to the next vehicle for destinations
        return solution.vehicles[vehicle].destinations[first_destination:num_customers + 1] + get_next_vehicles_destinations(solution, vehicle + 1, 1, remaining_destinations - ((num_customers + 1) - first_destination))
    else: # otherwise, the vehicle contains enough destinations between "first_destination" and the end of its list of destinations
        return solution.vehicles[vehicle].destinations[first_destination:first_destination + remaining_destinations]

def set_next_vehicles_destinations(solution: Union[OmbukiSolution, MMOEASASolution], vehicle: int, first_destination: int, remaining_destinations: int, reversed_destinations: List[Destination]) -> None:
    # most of the logic here is similar to "get_next_vehicles_destinations", the only difference being that, in this function, the nodes are being inserted instead of acquired
    if not (remaining_destinations and reversed_destinations):
        return
    num_customers = solution.vehicles[vehicle].get_num_of_customers_visited()
    if num_customers < first_destination + remaining_destinations:
        num_customers_inclusive = (num_customers + 1) - first_destination # list slicing is exclusive of the end point (meaning it would end at num_customers - 1), so + 1 will fix the exclusion
        solution.vehicles[vehicle].destinations[first_destination:num_customers + 1] = reversed_destinations[:num_customers_inclusive]
        del reversed_destinations[:num_customers_inclusive]
        set_next_vehicles_destinations(solution, vehicle + 1, 1, remaining_destinations - num_customers_inclusive, reversed_destinations)
    else:
        solution.vehicles[vehicle].destinations[first_destination:first_destination + remaining_destinations] = reversed_destinations
        reversed_destinations.clear()

def mutation(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution]) -> Union[OmbukiSolution, MMOEASASolution]:
    num_nodes_to_swap = rand(2, MUTATION_REVERSAL_LENGTH)
    first_reversal_node = rand(1, (len(instance.nodes) - 1) - num_nodes_to_swap)

    vehicle_num = -1
    num_destinations_tracker = 0
    for i, vehicle in enumerate(solution.vehicles): # because the mutation operator considers the routes as one collective chromosome (list of destinations from 1 to n, excluding starts and ends at the depot), we need to find which vehicle the position "first_reversal_node" belongs to if the solution were a chromosome
        if not num_destinations_tracker + vehicle.get_num_of_customers_visited() > first_reversal_node: # as soon as the sum of destinations is greater than "first_reversal_node", we've arrived at the vehicle where reversal should start
            num_destinations_tracker += vehicle.get_num_of_customers_visited()
        else:
            vehicle_num = i
            break

    first_destination = (first_reversal_node - num_destinations_tracker) + 1 # get the position of the "first_reversal_node" in the vehicle; + 1 to discount the depot at index 0 in the vehicle's destinations

    # the reason that the get and set functions are called recursively is because the mutation operator specified by Ombuki can swap customers across vehicles
    # therefore, the first call of the recursive functions can get/set the first one/two customers from one vehicle, then any remaining customers in the next vehicle
    reversed_destinations = get_next_vehicles_destinations(solution, vehicle_num, first_destination, num_nodes_to_swap)
    reversed_destinations = list(reversed(reversed_destinations))
    set_next_vehicles_destinations(solution, vehicle_num, first_destination, num_nodes_to_swap, reversed_destinations)

    solution.vehicles[vehicle_num].calculate_vehicle_load()
    solution.vehicles[vehicle_num].calculate_destinations_time_windows(instance)
    solution.vehicles[vehicle_num].calculate_length_of_route(instance)
    solution.objective_function(instance)
    return solution"""

def mutation(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution]) -> Union[OmbukiSolution, MMOEASASolution]:
    num_nodes_to_swap = rand(2, MUTATION_REVERSAL_LENGTH)
    vehicle = choice(list(filter(lambda v: v.get_num_of_customers_visited() >= num_nodes_to_swap, solution.vehicles)))
    first_reversal_node = rand(1, vehicle.get_num_of_customers_visited() - (num_nodes_to_swap - 1))

    vehicle.destinations[first_reversal_node : first_reversal_node + num_nodes_to_swap] = list(reversed(vehicle.destinations[first_reversal_node : first_reversal_node + num_nodes_to_swap]))

    vehicle.calculate_destinations_time_windows(instance, start_from=first_reversal_node)
    vehicle.calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution