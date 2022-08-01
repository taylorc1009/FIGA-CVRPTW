import copy
from random import shuffle
from typing import List, Set, Tuple
from FIGA.parameters import MUTATION_LONGEST_WAIT_PROBABILITY, MUTATION_LONGEST_ROUTE_PROBABILITY
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

    child_solution.calculate_routes_time_windows(instance)
    #child_solution.calculate_length_of_routes(instance) # this is not required here as the crossovers don't do any work with the total length of each route

    return child_solution

def SBCR_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicle: Vehicle) -> FIGASolution: # Single-child Best Cost Route Crossover
    crossover_solution = set_up_crossover_child(instance, parent_one, parent_two_vehicle)

    randomized_destinations = list(range(1, len(parent_two_vehicle.destinations) - 1))
    shuffle(randomized_destinations)
    for d in randomized_destinations:
        parent_destination = parent_two_vehicle.destinations[d]
        best_vehicle, best_position = instance.amount_of_vehicles, 1
        shortest_from_previous, shortest_to_next = (float(INT_MAX),) * 2
        highest_wait_time = 0.0
        #lowest_ready_time_difference = float(INT_MAX)
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
                    elif not found_feasible_location and crossover_solution.vehicles[i].destinations[j].wait_time > highest_wait_time:# and abs(crossover_solution.vehicles[i].destinations[j - 1].departure_time + distance_from_previous) < lowest_ready_time_difference:
                        # if no feasible insertion point has been found yet and the wait time of the previous destination is the highest that's been found, then record this as the best position
                        #lowest_ready_time_difference = abs(crossover_solution.vehicles[i].destinations[j - 1].departure_time + distance_from_previous)
                        best_vehicle, best_position, highest_wait_time = i, j, crossover_solution.vehicles[i].destinations[j].wait_time

        if not found_feasible_location and len(crossover_solution.vehicles) < instance.amount_of_vehicles:# and not best_vehicle < instance.amount_of_vehicles:
            best_vehicle = len(crossover_solution.vehicles)
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination))
        else:
            # best_vehicle and best_position will equal the insertion position before the vehicle with the longest wait time
            # that is if no feasible insertion point was found, otherwise it will equal the fittest feasible insertion point
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].calculate_vehicle_load()
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    crossover_solution.objective_function(instance)
    return crossover_solution

def ES_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicle: FIGASolution) -> FIGASolution: # Eliminate and Substitute Crossover
    crossover_solution = set_up_crossover_child(instance, parent_one, parent_two_vehicle)

    crossover_solution.vehicles.append(parent_two_vehicle)

    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.objective_function(instance)
    return crossover_solution

def select_random_vehicle(solution: FIGASolution, customers_required: int=2, exclude_values: Set[int]=None) -> int:
    if exclude_values is None:
        exclude_values = set()
    random_vehicle = -1
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
                total_wait = sum(destination.wait_time for destination in vehicle.get_customers_visited())

                if total_wait > longest_total_wait:
                    longest_waiting_vehicle = v
                    longest_total_wait = total_wait

    # check if not >= 0 instead of using "else" in case no vehicle has a wait time; this will never be the case, but this is here to be safe
    return longest_waiting_vehicle if longest_waiting_vehicle >= 0 else select_random_vehicle(solution)

"""def TWBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Sort Mutator
    longest_waiting_vehicle = select_route_with_longest_wait(solution)

    # sort all destinations between 1 and n - 1 by ready_time (exclude 1 and n - 1 as they're the depot nodes)
    solution.vehicles[longest_waiting_vehicle].destinations[1:-1] = sorted(solution.vehicles[longest_waiting_vehicle].get_customers_visited(), key=lambda d: d.node.ready_time)

    solution.vehicles[longest_waiting_vehicle].calculate_destinations_time_windows(instance)
    solution.vehicles[longest_waiting_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution"""

def swap(l1: List, index_one: int, index_two: int, l2: List=None) -> None:
    if l2:
        l1[index_one], l2[index_two] = l2[index_two], l1[index_one]
    else:
        l1[index_one], l1[index_two] = l1[index_two], l1[index_one]

def TWBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Swap Mutator
    longest_waiting_vehicle = select_route_with_longest_wait(solution)

    for d in range(1, solution.vehicles[longest_waiting_vehicle].get_num_of_customers_visited()):
        if solution.vehicles[longest_waiting_vehicle].destinations[d].node.ready_time > solution.vehicles[longest_waiting_vehicle].destinations[d + 1].node.ready_time:
            swap(solution.vehicles[longest_waiting_vehicle].destinations, d, d + 1)
            break

    solution.vehicles[longest_waiting_vehicle].calculate_destinations_time_windows(instance)
    solution.vehicles[longest_waiting_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def get_far_traveling_vehicle(solution: FIGASolution, skip_vehicles: Set[int]=None) -> int:
    if skip_vehicles is None:
        skip_vehicles = set()
    longest_route_length = 0
    furthest_traveling_vehicle = int() # this function will always assign a value to this variable

    if rand(1, 100) < MUTATION_LONGEST_ROUTE_PROBABILITY: # probability of selecting either the furthest traveling vehicle or a random vehicle
        # find the furthest traveling vehicle
        for v, vehicle in enumerate(solution.vehicles):
            if v not in skip_vehicles and vehicle.route_distance > longest_route_length and vehicle.get_num_of_customers_visited() > 2:
                furthest_traveling_vehicle = v
                longest_route_length = vehicle.route_distance
    if not furthest_traveling_vehicle >= 0:
        furthest_traveling_vehicle = select_random_vehicle(solution, customers_required=3)

    return furthest_traveling_vehicle

def DBT_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Distance-based Transfer Mutator
    first_furthest_traveling_vehicle = get_far_traveling_vehicle(solution)
    second_furthest_traveling_vehicle = get_far_traveling_vehicle(solution, skip_vehicles={first_furthest_traveling_vehicle})

    first_vehicle, second_vehicle = solution.vehicles[first_furthest_traveling_vehicle], solution.vehicles[second_furthest_traveling_vehicle]

    """for d in range(1, min(first_vehicle.get_num_of_customers_visited(), second_vehicle.get_num_of_customers_visited()) + 1):
        distance_from_previous = instance.get_distance(first_vehicle.destinations[d - 1].node.number, second_vehicle.destinations[d].node.number)
        
        simulated_arrival_time = first_vehicle.destinations[d - 1].departure_time + distance_from_previous
        if simulated_arrival_time < second_vehicle.destinations[d].node.ready_time:
            simulated_arrival_time = second_vehicle.destinations[d].node.ready_time
        simulated_departure_time = simulated_arrival_time + second_vehicle.destinations[d].node.service_duration
        
        if instance.get_distance(first_vehicle.destinations[d - 1].node.number, second_vehicle.destinations[d].node.number) < instance.get_distance(first_vehicle.destinations[d - 1].node.number, first_vehicle.destinations[d].node.number) \
            and simulated_departure_time + instance.get_distance(second_vehicle.destinations[d].node.number, first_vehicle.destinations[d].node.number) < first_vehicle.destinations[d].node.due_date:
            first_vehicle.destinations.insert(d, second_vehicle.destinations.pop(d))
            if not second_vehicle.get_num_of_customers_visited():
                del solution.vehicles[second_furthest_traveling_vehicle]
            
            first_vehicle.calculate_length_of_route(instance)
            first_vehicle.calculate_destinations_time_windows(instance)
            first_vehicle.calculate_vehicle_load()
            second_vehicle.calculate_length_of_route(instance)
            second_vehicle.calculate_destinations_time_windows(instance)
            second_vehicle.calculate_vehicle_load()
            solution.objective_function(instance)
            break"""

    for d1 in range(1, first_vehicle.get_num_of_customers_visited()):
        for d2 in range(1, second_vehicle.get_num_of_customers_visited()):
            first_distance_from_previous = instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, second_vehicle.destinations[d2].node.number)
            second_distance_from_previous = instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, first_vehicle.destinations[d1].node.number)
            
            first_simulated_arrival_time = first_vehicle.destinations[d1 - 1].departure_time + first_distance_from_previous
            if first_simulated_arrival_time < second_vehicle.destinations[d2].node.ready_time:
                first_simulated_arrival_time = second_vehicle.destinations[d2].node.ready_time
            first_simulated_departure_time = first_simulated_arrival_time + second_vehicle.destinations[d2].node.service_duration
            
            second_simulated_arrival_time = second_vehicle.destinations[d2 - 1].departure_time + second_distance_from_previous
            if second_simulated_arrival_time < first_vehicle.destinations[d1].node.ready_time:
                second_simulated_arrival_time = first_vehicle.destinations[d1].node.ready_time
            second_simulated_departure_time = second_simulated_arrival_time + first_vehicle.destinations[d1].node.service_duration

            if instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, second_vehicle.destinations[d2].node.number) <= instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, first_vehicle.destinations[d1].node.number) \
                and instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, first_vehicle.destinations[d1].node.number) <= instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, second_vehicle.destinations[d2].node.number) \
                and first_simulated_departure_time + instance.get_distance(second_vehicle.destinations[d2].node.number, first_vehicle.destinations[d1 + 1].node.number) < first_vehicle.destinations[d1 + 1].node.due_date \
                and second_simulated_departure_time + instance.get_distance(first_vehicle.destinations[d1].node.number, second_vehicle.destinations[d2 + 1].node.number) < second_vehicle.destinations[d2 + 1].node.due_date:
                swap(first_vehicle.destinations, d1, d2, l2=second_vehicle.destinations)
                
                first_vehicle.calculate_length_of_route(instance)
                first_vehicle.calculate_destinations_time_windows(instance)
                first_vehicle.calculate_vehicle_load()
                second_vehicle.calculate_length_of_route(instance)
                second_vehicle.calculate_destinations_time_windows(instance)
                second_vehicle.calculate_vehicle_load()
                solution.objective_function(instance)
                break

    return solution

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

def TWBLC_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Local Crossover Mutator
    random_vehicle = solution.vehicles[rand(0, len(solution.vehicles) - 1)]
    best_vehicle, best_position = None, None # will always be given a value as it's practically impossible to arrive at every destination exactly when their time windows open

    # best point from one vehicle would be where the arrival time is nearest a destination's ready_time
    best_ready_time_difference = INT_MAX # the best position would have a very small difference between the arrival time and the destination's ready_time
    for v, vehicle in enumerate(solution.vehicles):
        if vehicle is not random_vehicle:
            for destination_of_random in random_vehicle.get_customers_visited():
                for d, destination in enumerate(vehicle.get_customers_visited(), 1): # don't start from the leave-depot node (0) as then we would just be swapping the entire route (instead of crossing it over) if that point is the best fit
                    arrival_time = destination.departure_time + instance.get_distance(destination.node.number, destination_of_random.node.number)
                    ready_time_difference = abs(destination_of_random.node.ready_time - arrival_time)
                    if arrival_time <= destination_of_random.node.due_date and ready_time_difference < best_ready_time_difference:
                        #if ready_time_difference >= best_ready_time_difference: # for performance: theoretically, when the best point has been found, the difference of the current iteration will be higher the difference of the best 
                        #    break
                        best_ready_time_difference, best_vehicle, best_position = ready_time_difference, v, d

    if best_vehicle is not None and best_position is not None:
        for d, destination in enumerate(random_vehicle.get_customers_visited(), 1):
            if destination.departure_time + instance.get_distance(destination.node.number, solution.vehicles[best_vehicle].destinations[best_position].node.number) < solution.vehicles[best_vehicle].destinations[best_position].node.due_date:
                # slice the randomly selected vehicle's and the best-fitting vehicle's destinations lists from both points where it's feasible to cross them over, then crossover
                random_vehicle.destinations[d:-1], solution.vehicles[best_vehicle].destinations[best_position:-1] = solution.vehicles[best_vehicle].destinations[best_position:-1], random_vehicle.destinations[d:-1]
                break

        solution.calculate_length_of_routes(instance)
        solution.calculate_vehicles_loads()
        solution.calculate_routes_time_windows(instance)
        solution.objective_function(instance)

    return solution

def find_time_window_threatened_position(solution: FIGASolution) -> Tuple[int, int]:
    worst_route, worst_position, riskiest_difference = None, None, INT_MAX

    # find an infeasible route; if none are infeasible, select the destination with the smallest difference between its arrival time and its due date
    for v, vehicle in enumerate(solution.vehicles):
        for d, destination in enumerate(vehicle.get_customers_visited(), 1):
            if destination.arrival_time > destination.node.due_date:
                return v, d # any infeasible destination is bad, so return any
            elif destination.node.due_date - destination.arrival_time < riskiest_difference: # the difference (minus calculation) will never be negative if the arrival time is not greater than the due date
                worst_route, worst_position, riskiest_difference = v, d, destination.node.due_date - destination.arrival_time

    return worst_route, worst_position

def ATBR_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Arrival-Time-based Reorder Mutator
    worst_route, worst_position = find_time_window_threatened_position(solution)
    longest_waiting_vehicle, longest_waiting_position, longest_wait_time = None, None, 0

    # find the destination with the longest wait time and...
    for v, vehicle in enumerate(solution.vehicles):
        if v != worst_route:
            for d, destination in enumerate(vehicle.get_customers_visited(), 1):
                if destination.wait_time > longest_wait_time and vehicle.current_capacity + solution.vehicles[worst_route].destinations[worst_position].node.demand <= instance.capacity_of_vehicles:
                    longest_waiting_vehicle, longest_waiting_position, longest_wait_time = v, d, destination.wait_time

    # ... move the most "time-window-threatened" destination before it
    if longest_waiting_vehicle is not None and longest_waiting_position is not None:
        solution.vehicles[longest_waiting_vehicle].destinations.insert(longest_waiting_position, solution.vehicles[worst_route].destinations.pop(worst_position))

    solution.calculate_length_of_routes(instance)
    solution.calculate_vehicles_loads()
    solution.calculate_routes_time_windows(instance)
    solution.objective_function(instance)

    return solution

def try_feasible_swap(instance: ProblemInstance, solution: FIGASolution, first_vehicle: Vehicle, second_vehicle: Vehicle) -> None:
    for d1, destination_one in enumerate(first_vehicle.get_customers_visited(), 1):
        for d2, destination_two in enumerate(second_vehicle.get_customers_visited(), 1):
            distance_from_first_previous = instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, destination_two.node.number)
            distance_from_second_previous = instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, destination_one.node.number)

            first_simulated_arrival_time = first_vehicle.destinations[d1 - 1].departure_time + distance_from_first_previous
            if first_simulated_arrival_time > destination_two.node.due_date:
                continue
            if first_simulated_arrival_time < destination_two.node.ready_time:
                first_simulated_arrival_time = destination_two.node.ready_time
            first_simulated_departure_time = first_simulated_arrival_time + destination_two.node.service_duration
            
            second_simulated_arrival_time = second_vehicle.destinations[d2 - 1].departure_time + distance_from_second_previous
            if second_simulated_arrival_time > destination_one.node.due_date:
                continue
            if second_simulated_arrival_time < destination_one.node.ready_time:
                second_simulated_arrival_time = destination_one.node.ready_time
            second_simulated_departure_time = second_simulated_arrival_time + destination_one.node.service_duration

            if first_simulated_departure_time + instance.get_distance(destination_two.node.number, first_vehicle.destinations[d1 + 1].node.number) < first_vehicle.destinations[d1 + 1].node.due_date \
                and second_simulated_departure_time + instance.get_distance(destination_one.node.number, second_vehicle.destinations[d2 + 1].node.number) < second_vehicle.destinations[d2 + 1].node.due_date:
                swap(first_vehicle.destinations, d1, d2, l2=second_vehicle.destinations)
                first_vehicle.calculate_length_of_route(instance)
                first_vehicle.calculate_destinations_time_windows(instance)
                first_vehicle.calculate_vehicle_load()
                second_vehicle.calculate_length_of_route(instance)
                second_vehicle.calculate_destinations_time_windows(instance)
                second_vehicle.calculate_vehicle_load()
                solution.objective_function(instance)

                return

def FBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Feasibility-based Swap Mutator
    first_vehicle_index = select_random_vehicle(solution, customers_required=1)
    first_vehicle = solution.vehicles[first_vehicle_index]
    second_vehicle = solution.vehicles[select_random_vehicle(solution, customers_required=1, exclude_values=set({first_vehicle_index}))]

    try_feasible_swap(instance, solution, first_vehicle, second_vehicle)

    return solution
