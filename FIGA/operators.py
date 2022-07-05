import copy
from random import shuffle
from typing import List
from FIGA.constants import MUTATION_LONGEST_WAIT_PROBABILITY, MUTATION_LONGEST_ROUTE_PROBABILITY
from FIGA.figaSolution import FIGASolution
from common import INT_MAX, rand
from problemInstance import ProblemInstance
from vehicle import Vehicle

def set_up_crossover_child(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicle: Vehicle) -> FIGASolution:
    child_solution = copy.deepcopy(parent_one)

    nodes_to_remove = {d.node.number for d in parent_two_vehicle.get_customers_visited()} # create a set containing the numbers of every node in parent_two_vehicle to be merged into parent_one's routes
    i = 0
    while i < len(child_solution.vehicles) and nodes_to_remove:
        increment = True
        j = 1
        while j <= child_solution.vehicles[i].get_num_of_customers_visited() and nodes_to_remove:
            destination = child_solution.vehicles[i].destinations[j]

            if destination.node.number in nodes_to_remove:
                nodes_to_remove.remove(destination.node.number)
                child_solution.vehicles[i].current_capacity -= destination.node.demand

                if child_solution.vehicles[i].get_num_of_customers_visited() - 1 > 0:
                    del child_solution.vehicles[i].destinations[j]
                else:
                    increment = False
                    del child_solution.vehicles[i] # remove the vehicle if its route is empty
                    break # break, otherwise the while loop will start searching the next vehicle with "j" as the same value; without incrementing "i" and starting "j" at 0
            else: # only move to the next destination if "j" isn't the index of a destination to be removed
                j += 1
        if increment: # don't move to the next vehicle if an empty one was deleted
            i += 1

    child_solution.calculate_nodes_time_windows(instance)
    child_solution.calculate_vehicles_loads(instance)

    return child_solution

def crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicle: Vehicle) -> FIGASolution:
    crossover_solution = set_up_crossover_child(instance, parent_one, parent_two_vehicle)

    randomized_destinations = list(range(1, len(parent_two_vehicle.destinations) - 1))
    shuffle(randomized_destinations)
    for d in randomized_destinations:
        parent_destination = parent_two_vehicle.destinations[d]
        best_vehicle, best_position = instance.amount_of_vehicles, 1
        shortest_from_previous, shortest_to_next = (float(INT_MAX),) * 2
        highest_wait_time = 0.0
        found_feasible_location = False

        for i, vehicle in enumerate(crossover_solution.vehicles):
            if not vehicle.current_capacity + parent_destination.node.demand > instance.capacity_of_vehicles:
                for j in range(1, len(crossover_solution.vehicles[i].destinations)):
                    distance_from_previous = instance.get_distance(vehicle.destinations[j - 1].node.number, parent_destination.node.number)
                    distance_to_next = instance.get_distance(parent_destination.node.number, vehicle.destinations[j].node.number)

                    # used to simulate the time windows of the previous and next destinations to "parent_destination" if it were to be inserted into index j
                    # these are calculated so that we do not need to temporarily insert the parent_destination and then remove it again after evaluation of its fitness in that position
                    simulated_arrival_time = vehicle.destinations[j - 1].departure_time + distance_from_previous
                    if simulated_arrival_time < parent_destination.node.ready_time:
                        simulated_arrival_time = parent_destination.node.ready_time
                    simulated_departure_time = simulated_arrival_time + parent_destination.node.service_duration

                    # if, based on the simulated arrival and departure times, insertion does not violate time window constraints and the distance from the nodes at j - 1 and j is less than any that's been found, then record this as the best position
                    if not (simulated_arrival_time > parent_destination.node.due_date or simulated_departure_time + distance_to_next > vehicle.destinations[j].node.due_date) \
                            and ((distance_from_previous < shortest_from_previous and distance_to_next <= shortest_to_next) or (distance_from_previous <= shortest_from_previous and distance_to_next < shortest_to_next)):
                        best_vehicle, best_position, shortest_from_previous, shortest_to_next = i, j, distance_from_previous, distance_to_next
                        found_feasible_location = True
                    elif not found_feasible_location and crossover_solution.vehicles[i].destinations[j - 1].wait_time > highest_wait_time:
                        # if no feasible insertion point has been found yet and the wait time of the previous destination is the highest that's been found, then record this as the best position
                        best_vehicle, best_position, highest_wait_time = i, j - 1, crossover_solution.vehicles[i].destinations[j - 1].wait_time

        if not found_feasible_location and len(crossover_solution.vehicles) < instance.amount_of_vehicles:
            best_vehicle = len(crossover_solution.vehicles)
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination.node))
        else:
            # best_vehicle and best_position will equal the insertion position before the vehicle with the longest wait time
            # that is if no feasible insertion point was found, otherwise it will equal the fittest feasible insertion point
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].calculate_vehicle_load(instance)
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    crossover_solution.objective_function(instance)
    return crossover_solution

def select_random_vehicle(solution: FIGASolution, customers_required: int=2) -> int:
    random_vehicle = -1
    exclude_values = set()
    while not random_vehicle >= 0:
        random_vehicle = rand(0, len(solution.vehicles) - 1, exclude_values=exclude_values)
        if not solution.vehicles[random_vehicle].get_num_of_customers_visited() >= customers_required:
            exclude_values.add(random_vehicle)
            random_vehicle = -1
    return random_vehicle

def select_route_with_longest_wait(solution: FIGASolution) -> int:
    longest_waiting_vehicle = -1
    longest_total_wait = 0.0
    if rand(1, 100) < MUTATION_LONGEST_WAIT_PROBABILITY:
        for v, vehicle in enumerate(solution.vehicles):
            if vehicle.get_num_of_customers_visited() > 1:
                total_wait = 0.0

                for destination in vehicle.get_customers_visited():
                    total_wait += destination.wait_time

                    if total_wait > longest_total_wait:
                        longest_waiting_vehicle = v
                        longest_total_wait = total_wait

    # check if not >= 0 instead of using "else" in case no vehicle has a wait time; this will never be the case, but this is here to be safe
    return longest_waiting_vehicle if longest_waiting_vehicle >= 0 else select_random_vehicle(solution)

def TWBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Sort Mutator
    longest_waiting_vehicle = select_route_with_longest_wait(solution)

    # sort all destinations between 1 and n - 1 by ready_time (exclude 1 and n - 1 as they're the depot nodes)
    solution.vehicles[longest_waiting_vehicle].destinations[1:-1] = sorted(solution.vehicles[longest_waiting_vehicle].get_customers_visited(), key=lambda d: d.node.ready_time)

    solution.vehicles[longest_waiting_vehicle].calculate_destinations_time_windows(instance)
    solution.vehicles[longest_waiting_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def swap(l: List, index_one: int, index_two: int) -> None:
    l[index_one], l[index_two] = l[index_two], l[index_one]

def TWBSw_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Swap Mutator
    longest_waiting_vehicle = select_route_with_longest_wait(solution)

    for d in range(1, solution.vehicles[longest_waiting_vehicle].get_num_of_customers_visited()):
        if solution.vehicles[longest_waiting_vehicle].destinations[d].node.ready_time > solution.vehicles[longest_waiting_vehicle].destinations[d + 1].node.ready_time:
            swap(solution.vehicles[longest_waiting_vehicle].destinations, d, d + 1)
            break

    solution.vehicles[longest_waiting_vehicle].calculate_destinations_time_windows(instance)
    solution.vehicles[longest_waiting_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def swap_high_wait_time_destinations(instance: ProblemInstance, solution: FIGASolution, just_once: bool=False) -> FIGASolution:
    longest_waiting_vehicle = select_route_with_longest_wait(solution)

    destination = 1
    while destination < solution.vehicles[longest_waiting_vehicle].get_num_of_customers_visited():
        if solution.vehicles[longest_waiting_vehicle].destinations[destination].wait_time > solution.vehicles[longest_waiting_vehicle].destinations[destination + 1].wait_time:
            swap(solution.vehicles[longest_waiting_vehicle].destinations, destination, destination + 1)

            for _ in range(2): # fix the arrival and departure times for the nodes that were swapped
                solution.vehicles[longest_waiting_vehicle].calculate_destination_time_window(instance, destination - 1, destination)
                destination += 1
            if just_once:
                for d in range(destination, solution.vehicles[longest_waiting_vehicle].get_num_of_customers_visited() + 1): # fix the arrival and departure times of destinations that follow the swapped nodes
                    solution.vehicles[longest_waiting_vehicle].calculate_destination_time_window(instance, d - 1, d)
                break
            elif not destination > solution.vehicles[longest_waiting_vehicle].get_num_of_customers_visited(): # fix the arrival and departure time for only the destination after the node moved back one position
                solution.vehicles[longest_waiting_vehicle].calculate_destination_time_window(instance, destination, destination + 1)
        else:
            destination += 1

    solution.vehicles[longest_waiting_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def WTBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Wait-Time-based Swap Mutator
    return swap_high_wait_time_destinations(instance, solution)

def SWTBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Single Wait-Time-based Swap Mutator
    return swap_high_wait_time_destinations(instance, solution, just_once=True)

def swap_long_distance_destinations(instance: ProblemInstance, solution: FIGASolution, just_once: bool=False) -> FIGASolution:
    longest_route_length = 0
    furthest_travelling_vehicle = -1

    if rand(1, 100) < MUTATION_LONGEST_ROUTE_PROBABILITY:
        # find the furthest travelling vehicle
        for v, vehicle in enumerate(solution.vehicles):
            if vehicle.route_distance > longest_route_length and vehicle.get_num_of_customers_visited() > 2:
                furthest_travelling_vehicle = v
                longest_route_length = vehicle.route_distance
    if not furthest_travelling_vehicle >= 0:
        furthest_travelling_vehicle = select_random_vehicle(solution, customers_required=3)

    destination = 1
    while destination <= solution.vehicles[furthest_travelling_vehicle].get_num_of_customers_visited() - 2:
        # if the distance between "destination" and "destination + 1" is greater than "destination" and "destination + 2" then swap "destination + 1" and "destination + 2"
        if instance.get_distance(solution.vehicles[furthest_travelling_vehicle].destinations[destination].node.number, solution.vehicles[furthest_travelling_vehicle].destinations[destination + 1].node.number) > instance.get_distance(solution.vehicles[furthest_travelling_vehicle].destinations[destination].node.number, solution.vehicles[furthest_travelling_vehicle].destinations[destination + 2].node.number):
            swap(solution.vehicles[furthest_travelling_vehicle].destinations, destination + 1, destination + 2)
            if just_once:
                break
        destination += 1

    solution.vehicles[furthest_travelling_vehicle].calculate_length_of_route(instance)
    solution.vehicles[furthest_travelling_vehicle].calculate_destinations_time_windows(instance)
    solution.objective_function(instance)

    return solution

def DBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Distance-based Swap Mutator
    return swap_long_distance_destinations(instance, solution)

def SDBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Single Distance-based Swap Mutator
    return swap_long_distance_destinations(instance, solution, just_once=True)

def move_destination_to_fit_window(instance: ProblemInstance, solution: FIGASolution, reverse: bool=False) -> FIGASolution:
    random_vehicle = select_random_vehicle(solution)

    original_indexes = {destination.node.number: index for index, destination in enumerate(solution.vehicles[random_vehicle].get_customers_visited(), 1)} # will be used to get the current index of a destination to be moved forward or pushed back
    sorted_destinations = list(enumerate(sorted(solution.vehicles[random_vehicle].get_customers_visited(), key=lambda d: d.node.ready_time), 1)) # sort the destinations in a route by their ready_time
    if reverse: # if the list is reversed then we want to push the destination with the highest ready_time to the back of the route
        sorted_destinations = reversed(sorted_destinations)

    for d, destination in sorted_destinations:
        if destination.node.number != solution.vehicles[random_vehicle].destinations[d].node.number: # if the destination ("d") is not at the index that it should be in the sorted route, then move it from its current position to the index that it would be at in a sorted route
            solution.vehicles[random_vehicle].destinations.insert(d, solution.vehicles[random_vehicle].destinations.pop(original_indexes[destination.node.number]))
            break

    solution.vehicles[random_vehicle].calculate_destinations_time_windows(instance)
    solution.vehicles[random_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def TWBMF_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Move Forward Mutator
    return move_destination_to_fit_window(instance, solution)

def TWBPB_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Push-back Mutator
    return move_destination_to_fit_window(instance, solution, reverse=True)
