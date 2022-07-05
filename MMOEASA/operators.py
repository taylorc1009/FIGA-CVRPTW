import copy
from MMOEASA.constants import INFINITY
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.ombukiSolution import OmbukiSolution
from MMOEASA.auxiliaries import insert_unvisited_node
from common import rand
from problemInstance import ProblemInstance
from typing import List, Tuple, Set, Union
from vehicle import Vehicle

def move_destination(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution], vehicle_1: int, origin: int, vehicle_2: int, destination: int) -> Union[MMOEASASolution, OmbukiSolution]:
    # moves destinations from one position in one route to another position in another route, then calculates the statistics of both routes after this change
    solution.vehicles[vehicle_2].destinations.insert(destination, solution.vehicles[vehicle_1].destinations.pop(origin))

    if vehicle_1 != vehicle_2:
        solution.vehicles[vehicle_1].current_capacity -= solution.vehicles[vehicle_2].destinations[destination].node.demand
        solution.vehicles[vehicle_2].current_capacity += solution.vehicles[vehicle_2].destinations[destination].node.demand
        solution.vehicles[vehicle_2].calculate_destinations_time_windows(instance)
        solution.vehicles[vehicle_2].calculate_length_of_route(instance)

    solution.vehicles[vehicle_1].calculate_destinations_time_windows(instance)
    solution.vehicles[vehicle_1].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def get_random_vehicle(solution: Union[MMOEASASolution, OmbukiSolution], exclude_values: Set[int]=None, destinations_required: int=1) -> int:
    random_vehicle = rand(0, len(solution.vehicles) - 1, exclude_values=exclude_values)
    while solution.vehicles[random_vehicle].get_num_of_customers_visited() < destinations_required:
        random_vehicle = rand(0, len(solution.vehicles) - 1, exclude_values=exclude_values)
    return random_vehicle

def mutation1(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # move a random customer in a random vehicle from one position to another, random position in the same vehicle
    random_vehicle = get_random_vehicle(solution, destinations_required=2)
    num_customers = solution.vehicles[random_vehicle].get_num_of_customers_visited()
    origin_position = rand(1, num_customers)
    destination_position = rand(1, num_customers, exclude_values={origin_position})

    solution = move_destination(instance, solution, random_vehicle, origin_position, random_vehicle, destination_position)

    return solution

def mutation2(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # move a random customer in a random vehicle from one position to the fittest position in the same vehicle
    random_vehicle = get_random_vehicle(solution, destinations_required=2)
    num_customers = solution.vehicles[random_vehicle].get_num_of_customers_visited()
    origin_position = rand(1, num_customers)

    best_location, fitness_of_best_location = origin_position, INFINITY
    for i in range(1, num_customers + 1):
        if not i == origin_position:
            solution = move_destination(instance, solution, random_vehicle, origin_position, random_vehicle, i)

            if 0 <= solution.total_distance < fitness_of_best_location: # if the current position is fitter than the previous best, record it as the best position
                fitness_of_best_location = solution.total_distance
                best_location = i
            solution = move_destination(instance, solution, random_vehicle, i, random_vehicle, origin_position)

    if not best_location == origin_position:
        solution = move_destination(instance, solution, random_vehicle, origin_position, random_vehicle, best_location)

    return solution

def mutation3(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # move a random customer in a random vehicle from its original vehicle to another random position in another random vehicle
    random_origin_vehicle = get_random_vehicle(solution)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    random_destination_vehicle = get_random_vehicle(solution, exclude_values={random_origin_vehicle})
    destination_position = rand(1, solution.vehicles[random_destination_vehicle].get_num_of_customers_visited())

    solution = move_destination(instance, solution, random_origin_vehicle, origin_position, random_destination_vehicle, destination_position)

    if not solution.vehicles[random_origin_vehicle].get_num_of_customers_visited():
        del solution.vehicles[random_origin_vehicle]

    return solution

def mutation4(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # move a random customer in a random vehicle from its original vehicle to the fittest position in another random vehicle
    random_origin_vehicle = get_random_vehicle(solution)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    random_destination_vehicle = get_random_vehicle(solution, exclude_values={random_origin_vehicle})

    best_location, fitness_of_best_location = -1, INFINITY
    for i in range(1, solution.vehicles[random_destination_vehicle].get_num_of_customers_visited() + 1):
        solution = move_destination(instance, solution, random_origin_vehicle, origin_position, random_destination_vehicle, i)

        if 0 <= solution.total_distance < fitness_of_best_location: # if the current position is fitter than the previous best, record it as the best position
            fitness_of_best_location = solution.total_distance
            best_location = i
        solution = move_destination(instance, solution, random_destination_vehicle, i, random_origin_vehicle, origin_position)

    if best_location >= 0:
        solution = move_destination(instance, solution, random_origin_vehicle, origin_position, random_destination_vehicle, best_location)
        if not solution.vehicles[random_origin_vehicle].get_num_of_customers_visited():
            del solution.vehicles[random_origin_vehicle]

    return solution

def mutation5(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # select two random vehicles and random destinations from each of them, then swap the two selections
    random_origin_vehicle = get_random_vehicle(solution)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    random_destination_vehicle = get_random_vehicle(solution, exclude_values={random_origin_vehicle})
    destination_position = rand(1, solution.vehicles[random_destination_vehicle].get_num_of_customers_visited())

    solution = move_destination(instance, solution, random_origin_vehicle, origin_position, random_destination_vehicle, destination_position)
    solution = move_destination(instance, solution, random_destination_vehicle, destination_position + 1, random_origin_vehicle, origin_position)

    return solution

def mutation6(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # select a random vehicles and a random destination from it, then find another vehicle with the fittest insertion point
    random_origin_vehicle = get_random_vehicle(solution)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    best_vehicle, best_location, fitness_of_best_location = -1, -1, INFINITY
    for destination_vehicle in range(0, len(solution.vehicles)):
        if not random_origin_vehicle == destination_vehicle:
            num_customers = solution.vehicles[destination_vehicle].get_num_of_customers_visited()

            if num_customers > 0:
                for i in range(1, num_customers + 1):
                    solution = move_destination(instance, solution, random_origin_vehicle, origin_position, destination_vehicle, i)

                    if 0 <= solution.total_distance < fitness_of_best_location: # if the current position is fitter than the previous best, record it as the best position
                        fitness_of_best_location = solution.total_distance
                        best_vehicle = destination_vehicle
                        best_location = i
                    solution = move_destination(instance, solution, destination_vehicle, i, random_origin_vehicle, origin_position)

    if best_location >= 0:
        solution = move_destination(instance, solution, random_origin_vehicle, origin_position, best_vehicle, best_location)
        if not solution.vehicles[random_origin_vehicle].get_num_of_customers_visited():
            del solution.vehicles[random_origin_vehicle]

    return solution

def mutation7(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # select a random vehicle and random destination from it, then vehicle and its position with the lowest difference between the randomly selected destination's arrival time and make the swap
    random_origin_vehicle = get_random_vehicle(solution)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    best_vehicle, best_location, smallest_time_window_difference = -1, -1, INFINITY
    for destination_vehicle in range(0, len(solution.vehicles)):
        if not random_origin_vehicle == destination_vehicle:
            num_customers = solution.vehicles[destination_vehicle].get_num_of_customers_visited()

            if num_customers > 0:
                for i in range(1, num_customers + 2):
                    solution = move_destination(instance, solution, random_origin_vehicle, origin_position, destination_vehicle, i)
                    time_window_difference = abs(solution.vehicles[random_origin_vehicle].destinations[origin_position].arrival_time - solution.vehicles[destination_vehicle].destinations[i].arrival_time)

                    if solution.total_distance >= 0 and time_window_difference < smallest_time_window_difference: # if the current position has a nearer arrival time than the previous best, record it as the best position
                        smallest_time_window_difference = time_window_difference
                        best_vehicle = destination_vehicle
                        best_location = i
                    solution = move_destination(instance, solution, destination_vehicle, i, random_origin_vehicle, origin_position)

    if best_location >= 0:
        solution = move_destination(instance, solution, random_origin_vehicle, origin_position, best_vehicle, best_location)
        if not solution.vehicles[random_origin_vehicle].get_num_of_customers_visited():
            del solution.vehicles[random_origin_vehicle]

    return solution

def mutation8(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # select a random vehicle and a random position, then allocate the destination at the random position and any following it to a new vehicle
    if len(solution.vehicles) < instance.amount_of_vehicles:
        random_vehicle = get_random_vehicle(solution, destinations_required=2)
        origin_position = rand(1, solution.vehicles[random_vehicle].get_num_of_customers_visited())

        solution.vehicles.append(Vehicle.create_route(instance, solution.vehicles[random_vehicle].destinations[origin_position:-1]))

        if origin_position == 1:
            del solution.vehicles[random_vehicle]
        else:
            del solution.vehicles[random_vehicle].destinations[origin_position:-1]

        solution.calculate_nodes_time_windows(instance)
        solution.calculate_vehicles_loads(instance)
        solution.calculate_length_of_routes(instance)
        solution.objective_function(instance)

    return solution

def mutation9(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # select a random destination in a random vehicle and move it to a new route
    if len(solution.vehicles) < instance.amount_of_vehicles:
        random_vehicle = get_random_vehicle(solution, destinations_required=2)
        num_customers = solution.vehicles[random_vehicle].get_num_of_customers_visited()
        origin_position = rand(1, num_customers)

        solution.vehicles.append(Vehicle.create_route(instance, solution.vehicles[random_vehicle].destinations[origin_position].node))

        if num_customers == 1:
            del solution.vehicles[random_vehicle]
        else:
            del solution.vehicles[random_vehicle].destinations[origin_position]

        solution.calculate_nodes_time_windows(instance)
        solution.calculate_vehicles_loads(instance)
        solution.calculate_length_of_routes(instance)
        solution.objective_function(instance)

    return solution

def mutation10(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution]) -> Union[MMOEASASolution, OmbukiSolution]:
    # select a random vehicle and try to move all of its destinations into feasible positions in other vehicles; destinations that cannot be moved will remain in the original randomly selected vehicle
    random_origin_vehicle = get_random_vehicle(solution)

    infeasible_node_reallocations = 0
    for _ in range(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited() + 1):
        origin_position = 1 + infeasible_node_reallocations
        node_reallocated = False
        destination_vehicle = 0

        while destination_vehicle < len(solution.vehicles) and not node_reallocated:
            num_customers_destination = solution.vehicles[destination_vehicle].get_num_of_customers_visited()

            if not random_origin_vehicle == destination_vehicle and num_customers_destination >= 1:
                for destination_position in range(1, num_customers_destination + 1):
                    solution = move_destination(instance, solution, random_origin_vehicle, origin_position, destination_vehicle, destination_position)

                    if solution.total_distance > INFINITY / 2:
                        solution = move_destination(instance, solution, destination_vehicle, destination_position, random_origin_vehicle, origin_position)
                    else:
                        node_reallocated = True
                        break
            destination_vehicle += 1
        if not node_reallocated:
            infeasible_node_reallocations += 1

    if not infeasible_node_reallocations:
        del solution.vehicles[random_origin_vehicle]

    return solution

def vehicle_insertion_possible(unvisited_nodes: Set[int], new_vehicle: Vehicle) -> Tuple[bool, Set[int]]:
    nodes_to_insert = set([d.node.number for d in new_vehicle.get_customers_visited()]) # make a set containing the numbers of the nodes to be crossed over
    # the function "difference" returns the set "unvisited_nodes" without the values in "nodes_to_insert"
    # so, if the length of the set returned is equal to the length of "unvisited_nodes" minus the number of nodes to insert,
    # then every node can be inserted and insertion is possible
    return len(unvisited_nodes.difference(nodes_to_insert)) == len(unvisited_nodes) - len(nodes_to_insert), nodes_to_insert

def crossover1(instance: ProblemInstance, solution: Union[MMOEASASolution, OmbukiSolution], population: List[Union[MMOEASASolution, OmbukiSolution]], is_nondominated_set: bool) -> Union[MMOEASASolution, OmbukiSolution]:
    solution.vehicles = [v for v in solution.vehicles if rand(1, 100) < 50] # remove some routes from a solution; each vehicle is given a 50% of being removed
    random_solution = rand(0, len(population) - 1, exclude_values={solution.id} if not is_nondominated_set else {}) # if we're working with the non-dominated set then we don't need to prevent a solution from being randomly selected; the reason we would is so we don't crossover a solution from the population with itself
    unvisited_nodes = set(range(1, len(instance.nodes))).difference([d.node.number for v in solution.vehicles for d in v.get_customers_visited()]) # create a set containing the node numbers of every node, then remove every node' number that the vehicle with routes removed still visits

    for i in range(len(population[random_solution].vehicles)):
        if population[random_solution].vehicles[i].get_num_of_customers_visited() >= 1:
            insertion_possible, new_nodes = vehicle_insertion_possible(unvisited_nodes, population[random_solution].vehicles[i]) # a vehicle from the randomly selected solution's list will be inserted if all of its destinations are listed in the unvisited nodes

            if insertion_possible and len(solution.vehicles) < instance.amount_of_vehicles:
                solution.vehicles.append(copy.deepcopy(population[random_solution].vehicles[i]))
                unvisited_nodes.difference_update(new_nodes)

                if not unvisited_nodes:
                    break

    for node_number in unvisited_nodes: # will only perform any iterations if some destinations were not contained in any of the vehicles that were possible for crossover
        solution = insert_unvisited_node(solution, instance, node_number)
    solution.objective_function(instance)

    return solution
