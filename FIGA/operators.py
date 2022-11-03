import copy
from random import shuffle, choice, getrandbits
from typing import List, Set
from FIGA.parameters import MUTATION_MAX_SLICE_LENGTH, MUTATION_SHORT_ROUTE_POOL_SIZE, MUTATION_LONGEST_WAIT_PROBABILITY, MUTATION_LONGEST_ROUTE_PROBABILITY, MUTATION_MAX_FBS_SWAPS, MUTATION_MAX_LDHR_SWAPS, MUTATION_REVERSE_SWAP_PROBABILITY, MUTATION_ELIMINATE_SHORTEST_PROBABILITY
from FIGA.figaSolution import FIGASolution
from constants import INT_MAX
from common import rand
from destination import Destination
from problemInstance import ProblemInstance
from vehicle import Vehicle
from numpy import subtract

def set_up_crossover_child(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicles: List[Vehicle]) -> FIGASolution:
    child_solution = copy.deepcopy(parent_one)

    nodes_to_remove = set({d.node.number for vehicle in parent_two_vehicles for d in vehicle.get_customers_visited()}) # create a set containing the numbers of every node in parent_two_vehicle to be merged into parent_one's routes
    i = 0
    while i < len(child_solution.vehicles) and nodes_to_remove:
        increment = True
        j = 1
        while j <= child_solution.vehicles[i].get_num_of_customers_visited() and nodes_to_remove:
            destination = child_solution.vehicles[i].destinations[j]

            if destination.node.number in nodes_to_remove:
                nodes_to_remove.remove(destination.node.number)
                child_solution.vehicles[i].current_capacity -= destination.node.demand

                if child_solution.vehicles[i].get_num_of_customers_visited() > 1:
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
    child_solution.calculate_length_of_routes(instance)
    child_solution.objective_function(instance)

    return child_solution

def SBCR_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicles: List[Vehicle]) -> FIGASolution: # Single-child Best Cost Route Crossover
    crossover_solution = set_up_crossover_child(instance, parent_one, parent_two_vehicles)

    randomized_destinations = [destination for vehicle in parent_two_vehicles for destination in vehicle.get_customers_visited()]
    shuffle(randomized_destinations)
    for parent_destination in randomized_destinations:
        best_vehicle, best_position = (None,) * 2
        # best_vehicle_by_distance, best_position_by_distance = (None,) * 2
        # best_vehicle_by_time, best_position_by_time = (None,) * 2
        shortest_from_previous, shortest_to_next = (float(INT_MAX),) * 2
        highest_wait_time = 0.0
        #lowest_ready_time_difference = float(INT_MAX)
        found_feasible_location = False

        for i, vehicle in enumerate(crossover_solution.vehicles):
            if vehicle.current_capacity + parent_destination.node.demand <= instance.capacity_of_vehicles:
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
                        # ready_time_difference = abs(vehicle.destinations[j].node.ready_time - (crossover_solution.vehicles[i].destinations[j - 1].departure_time + distance_from_previous))
                        # if ready_time_difference < lowest_ready_time_difference:
                        #     best_vehicle_by_time, best_position_by_time, lowest_ready_time_difference = i, j, ready_time_difference
                        found_feasible_location = True
                    elif not found_feasible_location:
                        #ready_time_difference = abs(vehicle.destinations[j].node.ready_time - (crossover_solution.vehicles[i].destinations[j - 1].departure_time + distance_from_previous))
                        if crossover_solution.vehicles[i].destinations[j].wait_time > highest_wait_time:# and ready_time_difference < lowest_ready_time_difference:
                        # if no feasible insertion point has been found yet and the wait time of the previous destination is the highest that's been found, then record this as the best position
                            best_vehicle, best_position, highest_wait_time = i, j, crossover_solution.vehicles[i].destinations[j].wait_time

        # best_vehicle, best_position = (best_vehicle_by_time, best_position_by_time) \
        #     if not (best_vehicle_by_distance is not None and best_position_by_distance is not None) or (rand(0, 1) and best_vehicle_by_time is not None and best_position_by_time is not None) \
        #     else (best_vehicle_by_distance, best_position_by_distance)

        if not found_feasible_location and len(crossover_solution.vehicles) < instance.amount_of_vehicles and best_vehicle is None:
            best_vehicle, best_position = len(crossover_solution.vehicles), 1
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination)) # we don't need to give "Vehicle.create_route" a deep copy of the destination as it constructs an new Destination instance
        else:
            # best_vehicle and best_position will equal the insertion position before the vehicle with the longest wait time
            # that is if no feasible insertion point was found, otherwise it will equal the fittest feasible insertion point
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].current_capacity += parent_destination.node.demand
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance, start_from=best_position)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    """for parent_destination in randomized_destinations:
        best_vehicle, best_position = (None,) * 2
        shortest_total_distance = float(INT_MAX)
        highest_wait_time = 0.0
        found_feasible_location = False

        for i, vehicle in enumerate(crossover_solution.vehicles):
            if vehicle.current_capacity + parent_destination.node.demand <= instance.capacity_of_vehicles:
                for j in range(1, len(crossover_solution.vehicles[i].destinations)):
                    vehicle.destinations.insert(j, Destination(node=parent_destination.node))
                    
                    if vehicle.calculate_destinations_time_windows(instance, start_from=j):
                        vehicle.calculate_length_of_route(instance)
                        total_distance = sum(v.route_distance for v in crossover_solution.vehicles)
                        if total_distance < shortest_total_distance:
                            best_vehicle, best_position, shortest_total_distance = i, j, total_distance
                            found_feasible_location = True
                    elif not found_feasible_location:
                        if crossover_solution.vehicles[i].destinations[j].wait_time > highest_wait_time:
                            best_vehicle, best_position, highest_wait_time = i, j - 1, crossover_solution.vehicles[i].destinations[j].wait_time

                    vehicle.destinations.pop(j)
                    vehicle.calculate_destinations_time_windows(instance, start_from=j)

        if not found_feasible_location and len(crossover_solution.vehicles) < instance.amount_of_vehicles and best_vehicle is None:
            best_vehicle, best_position = len(crossover_solution.vehicles), 1
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination)) # we don't need to give "Vehicle.create_route" a deep copy of the destination as it constructs an new Destination instance
        else:
            # best_vehicle and best_position will equal the insertion position before the vehicle with the longest wait time
            # that is if no feasible insertion point was found, otherwise it will equal the fittest feasible insertion point
            try:
                crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))
            except:
                print("ERROR", found_feasible_location, len(crossover_solution.vehicles), best_vehicle)
                exit()

        crossover_solution.vehicles[best_vehicle].current_capacity += parent_destination.node.demand
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance, start_from=best_position)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)"""

    crossover_solution.objective_function(instance)
    return crossover_solution

def FBR_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicles: List[Vehicle]) -> FIGASolution: # Feasibility-based Best Route Crossover
    crossover_solution = set_up_crossover_child(instance, parent_one, parent_two_vehicles)
    randomized_destinations = [destination for vehicle in parent_two_vehicles for destination in vehicle.get_customers_visited()]
    
    shuffle(randomized_destinations)
    for parent_destination in randomized_destinations:
        best_vehicle, best_position = (None,) * 2
        highest_wait_time = 0.0
        #lowest_ready_time_difference = float(INT_MAX)
        feasible_locations = []

        for i, vehicle in enumerate(crossover_solution.vehicles):
            if vehicle.current_capacity + parent_destination.node.demand <= instance.capacity_of_vehicles:
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
                    if not (simulated_arrival_time > parent_destination.node.due_date or simulated_departure_time + distance_to_next > vehicle.destinations[j].node.due_date):
                        feasible_locations.append((i, j))
                    elif not feasible_locations:
                        #ready_time_difference = abs(vehicle.destinations[j].node.ready_time - (crossover_solution.vehicles[i].destinations[j - 1].departure_time + distance_from_previous))
                        if crossover_solution.vehicles[i].destinations[j].wait_time > highest_wait_time:# and ready_time_difference < lowest_ready_time_difference:
                        # if no feasible insertion point has been found yet and the wait time of the previous destination is the highest that's been found, then record this as the best position
                            best_vehicle, best_position, highest_wait_time = i, j, crossover_solution.vehicles[i].destinations[j].wait_time

        if feasible_locations:
            best_vehicle, best_position = choice(feasible_locations)
        if not feasible_locations and len(crossover_solution.vehicles) < instance.amount_of_vehicles and best_vehicle is None:
            best_vehicle, best_position = len(crossover_solution.vehicles), 1
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination)) # we don't need to give "Vehicle.create_route" a deep copy of the destination as it constructs an new Destination instance
        else:
            # best_vehicle and best_position will equal the insertion position before the vehicle with the longest wait time
            # that is if no feasible insertion point was found, otherwise it will equal the fittest feasible insertion point
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].current_capacity += parent_destination.node.demand
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance, start_from=best_position)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    crossover_solution.objective_function(instance)
    return crossover_solution

"""def ES_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two_vehicles: List[Vehicle]) -> FIGASolution: # Eliminate and Substitute Crossover
    crossover_solution = set_up_crossover_child(instance, parent_one, parent_two_vehicles)

    crossover_solution.vehicles += [copy.deepcopy(vehicle) for vehicle in parent_two_vehicles]

    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.objective_function(instance)
    return crossover_solution"""

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

"""def PMX_reinsertion(instance: ProblemInstance, crossover_solution: FIGASolution, node_number: int, nodes_being_replaced: List[int]) -> int:
    for vehicle in crossover_solution.vehicles:
        for destination in vehicle.get_customers_visited():
            if destination.node.number == node_number:
                destination.node = instance.nodes[nodes_being_replaced.pop(0)]
                return

def PM_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution) -> FIGASolution: # Partially Mapped Crossover
    crossover_solution = copy.deepcopy(parent_one)
    previous_destination_vehicle = None # prevents the same vehicle from being crossover twice, in order to avoid reverting the previous mapping

    map_length = rand(2, 5)
    origin_vehicle = parent_two.vehicles[select_random_vehicle(parent_two, customers_required=map_length)]
    previous_destination_vehicle = destination_vehicle_index = select_random_vehicle(crossover_solution, customers_required=(map_length if origin_vehicle.get_num_of_customers_visited() > map_length else map_length + 1), exclude_values=previous_destination_vehicle)
    destination_vehicle = crossover_solution.vehicles[destination_vehicle_index]
    cut_start = rand(1, min(destination_vehicle.get_num_of_customers_visited(), origin_vehicle.get_num_of_customers_visited()) - (map_length - 1))

    nodes_being_replaced, nodes_being_inserted = [d.node.number for d in destination_vehicle.destinations[cut_start : cut_start + map_length]], [d.node.number for d in origin_vehicle.destinations[cut_start : cut_start + map_length]]

    destination_vehicle.destinations[cut_start : cut_start + map_length] = [copy.deepcopy(destination) for destination in origin_vehicle.destinations[cut_start : cut_start + map_length]]
    for node_number in nodes_being_inserted:
        if node_number not in nodes_being_replaced:
            PMX_reinsertion(instance, crossover_solution, node_number, nodes_being_replaced)
        else:
            nodes_being_replaced.remove(node_number)

    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.calculate_vehicles_loads()
    crossover_solution.calculate_routes_time_windows(instance)
    crossover_solution.objective_function(instance)

    return crossover_solution"""

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

def swap(instance: ProblemInstance, vehicle_one: Vehicle, index_one: int, index_two: int, vehicle_two: Vehicle=None) -> bool:
    if vehicle_two:
        vehicle_one.destinations[index_one], vehicle_two.destinations[index_two] = vehicle_two.destinations[index_two], vehicle_one.destinations[index_one]
        vehicle_one.current_capacity = (vehicle_one.current_capacity - vehicle_two.destinations[index_two].node.demand) + vehicle_one.destinations[index_one].node.demand
        vehicle_two.current_capacity = (vehicle_two.current_capacity - vehicle_one.destinations[index_one].node.demand) + vehicle_two.destinations[index_two].node.demand
        return vehicle_one.calculate_destinations_time_windows(instance, start_from=index_one) and vehicle_one.current_capacity <= instance.capacity_of_vehicles and vehicle_two.calculate_destinations_time_windows(instance, start_from=index_two) and vehicle_two.current_capacity <= instance.capacity_of_vehicles
    else:
        vehicle_one.destinations[index_one], vehicle_one.destinations[index_two] = vehicle_one.destinations[index_two], vehicle_one.destinations[index_one]
        return vehicle_one.calculate_destinations_time_windows(instance, start_from=min(index_one, index_two)) # in the use cases of this "else" block, index_one will always be less than index_two

def TWBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Swap Mutator
    vehicle = solution.vehicles[select_route_with_longest_wait(solution)]
    reverse = rand(1, 100) < MUTATION_REVERSE_SWAP_PROBABILITY
    possible_swaps = []

    for d in range(1, vehicle.get_num_of_customers_visited()):
        if (not reverse and vehicle.destinations[d].node.ready_time > vehicle.destinations[d + 1].node.ready_time) \
            or (reverse and vehicle.destinations[d].node.ready_time < vehicle.destinations[d + 1].node.ready_time):
            possible_swaps.append(d)

    if not possible_swaps:
        return None

    destination = choice(possible_swaps)
    swap(instance, vehicle, destination, destination + 1)
    vehicle.calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def get_far_traveling_vehicle(solution: FIGASolution, skip_vehicles: Set[int]=None) -> int:
    if skip_vehicles is None:
        skip_vehicles = set()
    longest_route_length = 0
    furthest_traveling_vehicle = None # this function will always assign a value to this variable

    if rand(1, 100) < MUTATION_LONGEST_ROUTE_PROBABILITY: # probability of selecting either the furthest traveling vehicle or a random vehicle
        # find the furthest traveling vehicle
        for v, vehicle in enumerate(solution.vehicles):
            if v not in skip_vehicles and vehicle.route_distance > longest_route_length and vehicle.get_num_of_customers_visited() > 2:
                furthest_traveling_vehicle = v
                longest_route_length = vehicle.route_distance
    if furthest_traveling_vehicle is None:
        furthest_traveling_vehicle = select_random_vehicle(solution, customers_required=3, exclude_values=skip_vehicles)

    return furthest_traveling_vehicle

def DBT_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution | None: # Distance-based Transfer Mutator
    first_vehicle_index = select_random_vehicle(solution)
    second_vehicle_index = get_far_traveling_vehicle(solution, skip_vehicles={first_vehicle_index})
    first_vehicle, second_vehicle = solution.vehicles[first_vehicle_index], solution.vehicles[second_vehicle_index]
    feasible_locations = []

    for d1, first_destination in enumerate(first_vehicle.get_customers_visited(), 1):
        for d2, second_destination in enumerate(second_vehicle.get_customers_visited(), 1):
            if instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, second_destination.node.number) < instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, first_destination.node.number) \
                and first_vehicle.current_capacity + second_destination.node.demand <= instance.capacity_of_vehicles:
                first_vehicle.destinations.insert(d1, copy.deepcopy(second_destination))

                if first_vehicle.calculate_destinations_time_windows(instance, start_from=d1):
                    feasible_locations.append((d1, d2))
                first_vehicle.destinations.pop(d1)
                first_vehicle.calculate_destinations_time_windows(instance, start_from=d1)

    if not feasible_locations:
        return None

    first_position, second_position = choice(feasible_locations)
    first_vehicle.destinations.insert(first_position, copy.deepcopy(second_vehicle.destinations[second_position]))
    first_vehicle.calculate_destinations_time_windows(instance, start_from=first_position)
    if second_vehicle.get_num_of_customers_visited() == 1:
        del solution.vehicles[second_vehicle_index]
    else:
        second_vehicle.destinations.pop(second_position)
        second_vehicle.current_capacity -= first_vehicle.destinations[first_position].node.demand
        second_vehicle.calculate_length_of_route(instance)
        second_vehicle.calculate_destinations_time_windows(instance, start_from=second_position)

    first_vehicle.current_capacity += first_vehicle.destinations[first_position].node.demand
    first_vehicle.calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def DBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution | None: # Distance-based Swap Mutator
    first_vehicle_index = select_random_vehicle(solution)
    second_vehicle_index = get_far_traveling_vehicle(solution, skip_vehicles={first_vehicle_index})
    first_vehicle, second_vehicle = solution.vehicles[first_vehicle_index], solution.vehicles[second_vehicle_index]
    feasible_locations = []

    for d1 in range(1, first_vehicle.get_num_of_customers_visited()):
        for d2 in range(1, second_vehicle.get_num_of_customers_visited()):
            if first_vehicle.current_capacity + second_vehicle.destinations[d2].node.demand <= instance.capacity_of_vehicles and second_vehicle.current_capacity + first_vehicle.destinations[d1].node.demand <= instance.capacity_of_vehicles:
                first_distance_from_previous = instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, second_vehicle.destinations[d2].node.number)
                second_distance_from_previous = instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, first_vehicle.destinations[d1].node.number)

                if first_distance_from_previous <= instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, first_vehicle.destinations[d1].node.number) \
                    and second_distance_from_previous <= instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, second_vehicle.destinations[d2].node.number):

                    if swap(instance, first_vehicle, d1, d2, vehicle_two=second_vehicle):
                        feasible_locations.append((d1, d2))
                        # first_vehicle.calculate_length_of_route(instance)
                        # second_vehicle.calculate_length_of_route(instance)
                        # solution.objective_function(instance)
                        # return solution
                    swap(instance, first_vehicle, d1, d2, vehicle_two=second_vehicle)

    if not feasible_locations:
        return None

    first_position, second_position = choice(feasible_locations)
    swap(instance, first_vehicle, first_position, second_position, vehicle_two=second_vehicle)
    first_vehicle.calculate_length_of_route(instance)
    second_vehicle.calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def try_distance_based_swap(instance: ProblemInstance, solution: FIGASolution, first_vehicle: Vehicle, second_vehicle: Vehicle) -> bool:
    feasible_locations = []

    for d1 in range(1, first_vehicle.get_num_of_customers_visited()):
        for d2 in range(1, second_vehicle.get_num_of_customers_visited()):
            if not (d1 == 1 and d2 == 1):
                if first_vehicle.current_capacity + second_vehicle.destinations[d2].node.demand <= instance.capacity_of_vehicles and second_vehicle.current_capacity + first_vehicle.destinations[d1].node.demand <= instance.capacity_of_vehicles:
                    first_distance_from_previous = instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, second_vehicle.destinations[d2].node.number)
                    second_distance_from_previous = instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, first_vehicle.destinations[d1].node.number)

                    if (((second_vehicle.destinations[d2].node.ready_time - first_vehicle.destinations[d1 - 1].node.ready_time) > (first_vehicle.destinations[d1].node.ready_time - first_vehicle.destinations[d1 - 1].node.ready_time) and first_distance_from_previous < instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, first_vehicle.destinations[d1].node.number)) \
                        #! using "and" in the next line, instead of "or", appears to cause greater convergence
                        or ((first_vehicle.destinations[d1].node.ready_time - second_vehicle.destinations[d2 - 1].node.ready_time) > (second_vehicle.destinations[d2].node.ready_time - second_vehicle.destinations[d2 - 1].node.ready_time) and second_distance_from_previous < instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, second_vehicle.destinations[d2].node.number))):

                        if swap(instance, first_vehicle, d1, d2, vehicle_two=second_vehicle):
                            feasible_locations.append((d1, d2))
                            # first_vehicle.calculate_length_of_route(instance)
                            # second_vehicle.calculate_length_of_route(instance)
                            # solution.objective_function(instance)
                            # return
                        swap(instance, first_vehicle, d1, d2, vehicle_two=second_vehicle)

    if not feasible_locations:
        return False

    first_position, second_position = choice(feasible_locations)

    swap(instance, first_vehicle, first_position, second_position, vehicle_two=second_vehicle)
    first_vehicle.calculate_length_of_route(instance)
    second_vehicle.calculate_length_of_route(instance)
    solution.objective_function(instance)

    return True

def LDHR_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution | None: # Low Distance and High Ready-time Mutator
    first_vehicle_index = select_random_vehicle(solution)
    first_vehicle = solution.vehicles[first_vehicle_index]
    
    remaining_vehicles = list(filter(lambda v: v is not first_vehicle, solution.vehicles))
    num_swaps, max_swaps = 0, rand(1, MUTATION_MAX_LDHR_SWAPS)

    while num_swaps < max_swaps and remaining_vehicles:
        second_vehicle = choice(remaining_vehicles)
        num_swaps += int(try_distance_based_swap(instance, solution, first_vehicle, second_vehicle))
        if num_swaps < max_swaps:
            remaining_vehicles.remove(second_vehicle)

    return solution if num_swaps > 0 else None

def TWBR_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Time-Window-based Reorder Mutator
    random_vehicle = select_random_vehicle(solution)

    original_indexes = {destination.node.number: index for index, destination in enumerate(solution.vehicles[random_vehicle].get_customers_visited(), 1)} # will be used to get the current index of a destination to be moved forward or pushed back
    sorted_destinations = list(enumerate(sorted(solution.vehicles[random_vehicle].get_customers_visited(), key=lambda d: d.node.ready_time), 1)) # sort the destinations in a route by their ready_time
    if bool(getrandbits(1)): # if the list is reversed then we want to push the destination with the highest ready_time to the back of the route
        sorted_destinations = reversed(sorted_destinations)

    for d, destination in sorted_destinations:
        if destination.node.number != solution.vehicles[random_vehicle].destinations[d].node.number: # if the destination ("d") is not at the index that it should be in the sorted route, then move it from its current position to the index that it would be at in a sorted route
            solution.vehicles[random_vehicle].destinations.insert(d, solution.vehicles[random_vehicle].destinations.pop(original_indexes[destination.node.number]))
            break

    solution.vehicles[random_vehicle].calculate_destinations_time_windows(instance)
    solution.vehicles[random_vehicle].calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution

def TWBLC_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution | None: # Time-Window-based Local Crossover Mutator
    origin_vehicle_index = select_random_vehicle(solution)
    origin_vehicle, destination_vehicle = solution.vehicles[origin_vehicle_index], solution.vehicles[select_random_vehicle(solution, exclude_values={origin_vehicle_index})]

    best_position = None # will always be given a value as it's practically impossible to arrive at every destination exactly when their time windows open
    best_ready_time_difference = INT_MAX # the best position would have a very small difference between the arrival time and the destination's ready_time

    # best point from one vehicle would be where the arrival time is nearest a destination's ready_time
    for destination_to_move in origin_vehicle.get_customers_visited():
        for d, destination in enumerate(destination_vehicle.get_customers_visited(), 1): # don't start from the leave-depot node (0) as then we would just be swapping the entire route (instead of crossing it over) if that point is the best fit
            arrival_time = destination.departure_time + instance.get_distance(destination.node.number, destination_to_move.node.number)
            ready_time_difference = abs(destination_to_move.node.ready_time - arrival_time)
            if arrival_time <= destination_to_move.node.due_date and ready_time_difference < best_ready_time_difference:
                #if ready_time_difference >= best_ready_time_difference: # for performance: theoretically, when the best point has been found, the difference of the current iteration will be higher the difference of the best 
                #    break
                best_ready_time_difference, best_position = ready_time_difference, d

    if best_position is None:
        return None

    for d, destination in enumerate(origin_vehicle.get_customers_visited(), 1):
        if destination.departure_time + instance.get_distance(destination.node.number, destination_vehicle.destinations[best_position].node.number) < destination_vehicle.destinations[best_position].node.due_date:
            # slice the randomly selected vehicle's and the best-fitting vehicle's destinations lists from both points where it's feasible to cross them over, then crossover
            origin_vehicle.destinations[d:], destination_vehicle.destinations[best_position:] = destination_vehicle.destinations[best_position:], origin_vehicle.destinations[d:]

            origin_vehicle.calculate_length_of_route(instance)
            origin_vehicle.calculate_vehicle_load()
            origin_vehicle.calculate_destinations_time_windows(instance, start_from=d)
            destination_vehicle.calculate_length_of_route(instance)
            destination_vehicle.calculate_vehicle_load()
            destination_vehicle.calculate_destinations_time_windows(instance, start_from=best_position)
            solution.objective_function(instance)

            return solution

"""def find_time_window_threatened_position(solution: FIGASolution) -> Tuple[int, int]:
    worst_route, worst_position, riskiest_difference = None, None, INT_MAX

    # find an infeasible route; if none are infeasible, select the destination with the smallest difference between its arrival time and its due date
    if rand(1, 100) < MUTATION_THREATENED_WINDOW_PROBABILITY:
        for v, vehicle in enumerate(solution.vehicles):
            for d, destination in enumerate(vehicle.get_customers_visited(), 1):
                if destination.arrival_time > destination.node.due_date:
                    return v, d # any infeasible destination is bad, so return any
                elif destination.node.due_date - destination.arrival_time < riskiest_difference: # the difference (minus calculation) will never be negative if the arrival time is not greater than the due date
                    worst_route, worst_position, riskiest_difference = v, d, destination.node.due_date - destination.arrival_time
    else:
        worst_route = select_random_vehicle(solution, customers_required=1)
        worst_position = rand(1, solution.vehicles[worst_route].get_num_of_customers_visited())

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

    return solution"""

def try_feasible_swap(instance: ProblemInstance, solution: FIGASolution, first_vehicle: Vehicle, second_vehicle: Vehicle) -> bool:
    feasible_locations = []

    for d1, destination_one in enumerate(first_vehicle.get_customers_visited(), 1):
        for d2, destination_two in enumerate(second_vehicle.get_customers_visited(), 1):
            if not (d1 == 1 and d2 == 1):
                first_arrival_time, second_arrival_time = first_vehicle.destinations[d1 - 1].departure_time + instance.get_distance(first_vehicle.destinations[d1 - 1].node.number, destination_two.node.number), second_vehicle.destinations[d2 - 1].departure_time + instance.get_distance(second_vehicle.destinations[d2 - 1].node.number, destination_one.node.number)
                if first_arrival_time <= destination_two.node.due_date and second_arrival_time <= destination_one.node.due_date:
                    if abs(destination_two.node.ready_time - first_arrival_time) < abs(destination_two.node.ready_time - destination_two.arrival_time) \
                    and abs(destination_one.node.ready_time - second_arrival_time) < abs(destination_one.node.ready_time - destination_one.arrival_time):
                        feasible_locations.append((d1, d2))
                else:
                    break

    if not feasible_locations:
        return False

    first_position, second_position = choice(feasible_locations)

    first_vehicle.destinations[first_position:], second_vehicle.destinations[second_position:] = second_vehicle.destinations[second_position:], first_vehicle.destinations[first_position:]
                    
    first_vehicle.calculate_length_of_route(instance)
    first_vehicle.calculate_destinations_time_windows(instance, start_from=first_position)
    first_vehicle.calculate_vehicle_load()
    second_vehicle.calculate_length_of_route(instance)
    second_vehicle.calculate_destinations_time_windows(instance, start_from=second_position)
    second_vehicle.calculate_vehicle_load()
    solution.objective_function(instance)

    return True

def FBS_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution: # Feasibility-based Swap Mutator
    first_vehicle = solution.vehicles[select_random_vehicle(solution)]

    remaining_vehicles = list(filter(lambda v: v is not first_vehicle, solution.vehicles))
    num_swaps, max_swaps = 0, rand(1, MUTATION_MAX_FBS_SWAPS)

    while num_swaps < max_swaps and remaining_vehicles:
        second_vehicle = choice(remaining_vehicles)
        num_swaps += int(try_feasible_swap(instance, solution, first_vehicle, second_vehicle))
        if num_swaps < max_swaps:
            remaining_vehicles.remove(second_vehicle)

    return solution if num_swaps > 0 else None

def select_short_route(solution: FIGASolution) -> Vehicle:
    vehicles = sorted(solution.vehicles, key=lambda v: v.get_num_of_customers_visited())
    return choice(vehicles[:min(MUTATION_SHORT_ROUTE_POOL_SIZE, len(vehicles) - 1)])

def try_feasible_reallocation(instance: ProblemInstance, solution: FIGASolution, random_origin_vehicle: Vehicle, origin_position: int) -> bool:
    shuffle(solution.vehicles)
    for destination_vehicle in solution.vehicles:
        if destination_vehicle is not random_origin_vehicle and destination_vehicle.current_capacity + random_origin_vehicle.destinations[origin_position].node.demand <= instance.capacity_of_vehicles:
            for destination_position in range(1, destination_vehicle.get_num_of_customers_visited() + 1):
                destination_vehicle.destinations.insert(destination_position, copy.deepcopy(random_origin_vehicle.destinations[origin_position]))
                feasible = destination_vehicle.calculate_destinations_time_windows(instance, start_from=destination_position)

                if feasible:
                    random_origin_vehicle.destinations.pop(origin_position)
                    destination_vehicle.current_capacity += destination_vehicle.destinations[destination_position].node.demand
                    return True
                else:
                    destination_vehicle.destinations.pop(destination_position)
                    destination_vehicle.calculate_destinations_time_windows(instance, start_from=destination_position)
    return False

def VE_mutation(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution | None: # Vehicle Elimination Mutator
    # select a random vehicle and try to move all of its destinations into feasible positions in other vehicles; destinations that cannot be moved will remain in the original randomly selected vehicle
    random_origin_vehicle = select_short_route(solution) if rand(1, 100) < MUTATION_ELIMINATE_SHORTEST_PROBABILITY else solution.vehicles[select_random_vehicle(solution, customers_required=1)]
    original_length = random_origin_vehicle.get_num_of_customers_visited()

    origin_position = 1
    while origin_position <= random_origin_vehicle.get_num_of_customers_visited():
        if not try_feasible_reallocation(instance, solution, random_origin_vehicle, origin_position):
            origin_position += 1

    if origin_position == 1:
        del solution.vehicles[solution.vehicles.index(random_origin_vehicle)] # this ".index()" is necessary because "try_feasible_reallocation" shuffles the list, so the index needs to be acquired after the list is shuffled
    elif random_origin_vehicle.get_num_of_customers_visited() == original_length:
        return None

    random_origin_vehicle.calculate_destinations_time_windows(instance)
    random_origin_vehicle.calculate_vehicle_load()

    solution.calculate_length_of_routes(instance)
    solution.objective_function(instance)

    return solution

def PBS_mutator(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution | None: # Partition-based Swap Mutator
    first_vehicle_index = select_random_vehicle(solution, customers_required=1)
    first_vehicle, second_vehicle = solution.vehicles[first_vehicle_index], solution.vehicles[select_random_vehicle(solution, customers_required=1, exclude_values={first_vehicle_index})]

    increment_switch = bool(getrandbits(1))
    max_length = rand(2, MUTATION_MAX_SLICE_LENGTH)
    slice_beginnings, slice_ends = (None,) * 2
    d1, d2 = (1,) * 2
    first_num_destinations, second_num_destinations = first_vehicle.get_num_of_customers_visited(), second_vehicle.get_num_of_customers_visited()
    feasible_slices = []

    while (not slice_beginnings and (d1 < first_num_destinations or d2 < second_num_destinations)) or (slice_beginnings and (d1 <= first_num_destinations or d2 <= second_num_destinations)):
        if not slice_beginnings:
            if swap(instance, first_vehicle, d1, d2, vehicle_two=second_vehicle):
                slice_beginnings, slice_ends = ((d1, d2),) * 2
            swap(instance, first_vehicle, d1, d2, vehicle_two=second_vehicle)
        else:
            ends_before = slice_ends
            first_beginning, second_beginning = slice_beginnings
            first_temp_end, second_temp_end = first_beginning + ((d1 + 1) - first_beginning), second_beginning + ((d2 + 1) - second_beginning)
            first_vehicle.destinations[first_beginning:first_temp_end], second_vehicle.destinations[second_beginning:second_temp_end] = second_vehicle.destinations[second_beginning:second_temp_end], first_vehicle.destinations[first_beginning:first_temp_end]
            first_feasibility, second_feasibility = first_vehicle.calculate_destinations_time_windows(instance, start_from=first_beginning), second_vehicle.calculate_destinations_time_windows(instance, start_from=second_beginning)
            first_vehicle.calculate_vehicle_load()
            second_vehicle.calculate_vehicle_load()
            
            first_temp_end, second_temp_end = first_beginning + ((d2 + 1) - second_beginning), second_beginning + ((d1 + 1) - first_beginning)
            if first_feasibility and second_feasibility and first_vehicle.current_capacity <= instance.capacity_of_vehicles and second_vehicle.current_capacity <= instance.capacity_of_vehicles and d1 < first_num_destinations and d2 < second_num_destinations:
                slice_ends = (d1, d2)
                d1 += 1
                d2 += 1
            elif first_feasibility and not second_feasibility:
                second_temp_end -= 1
                first_vehicle.destinations.insert(first_temp_end, second_vehicle.destinations.pop(second_temp_end))
                first_feasibility, second_feasibility = first_vehicle.calculate_destinations_time_windows(instance, start_from=first_beginning), second_vehicle.calculate_destinations_time_windows(instance, start_from=second_beginning)
                first_vehicle.calculate_vehicle_load()
                second_vehicle.calculate_vehicle_load()

                if first_feasibility and second_feasibility and first_vehicle.current_capacity <= instance.capacity_of_vehicles and second_vehicle.current_capacity <= instance.capacity_of_vehicles and d2 <= second_num_destinations:
                    slice_ends = (slice_ends[0], d2)
                    d2 += 1
            elif second_feasibility and not first_feasibility:
                first_temp_end -= 1
                second_vehicle.destinations.insert(second_temp_end, first_vehicle.destinations.pop(first_temp_end))
                first_feasibility, second_feasibility = first_vehicle.calculate_destinations_time_windows(instance, start_from=first_beginning), second_vehicle.calculate_destinations_time_windows(instance, start_from=second_beginning)
                first_vehicle.calculate_vehicle_load()
                second_vehicle.calculate_vehicle_load()

                if first_feasibility and second_feasibility and first_vehicle.current_capacity <= instance.capacity_of_vehicles and second_vehicle.current_capacity <= instance.capacity_of_vehicles and d1 <= first_num_destinations:
                    slice_ends = (d1, slice_ends[1])
                    d1 += 1

            if slice_ends != slice_beginnings and (slice_ends == ends_before or max_length in set(subtract(slice_ends, slice_beginnings)) or (not slice_ends == ends_before and d1 == first_num_destinations + 1 and d2 == second_num_destinations + 1)):
                feasible_slices.append((slice_beginnings, slice_ends))
                d1, d2 = slice_beginnings
                slice_beginnings, slice_ends = (None,) * 2
            elif slice_ends == slice_beginnings:
                slice_beginnings, slice_ends = (None,) * 2
            first_vehicle.destinations[first_beginning:first_temp_end], second_vehicle.destinations[second_beginning:second_temp_end] = second_vehicle.destinations[second_beginning:second_temp_end], first_vehicle.destinations[first_beginning:first_temp_end]

            first_vehicle.calculate_vehicle_load()
            first_vehicle.calculate_destinations_time_windows(instance, start_from=first_beginning)
            second_vehicle.calculate_vehicle_load()
            second_vehicle.calculate_destinations_time_windows(instance, start_from=second_beginning)

            if slice_beginnings and slice_ends != slice_beginnings:
                continue
        if (increment_switch or slice_beginnings) and d1 <= first_num_destinations:
            d1 += 1
        if (not increment_switch or slice_beginnings) and d2 <= second_num_destinations:
            d2 += 1
        if not (increment_switch and d2 > second_num_destinations) and not (not increment_switch and d1 > first_num_destinations):
            increment_switch = not increment_switch

    if not feasible_slices:
        return None

    slice_beginnings, slice_ends = choice(feasible_slices)
    first_beginning, second_beginning = slice_beginnings
    first_end, second_end = slice_ends

    first_vehicle.destinations[first_beginning:first_end + 1], second_vehicle.destinations[second_beginning:second_end + 1] = second_vehicle.destinations[second_beginning:second_end + 1], first_vehicle.destinations[first_beginning:first_end + 1]
    first_vehicle.calculate_destinations_time_windows(instance, start_from=first_beginning)
    first_vehicle.calculate_vehicle_load()
    first_vehicle.calculate_length_of_route(instance)
    second_vehicle.calculate_destinations_time_windows(instance, start_from=second_beginning)
    second_vehicle.calculate_vehicle_load()
    second_vehicle.calculate_length_of_route(instance)
    solution.objective_function(instance)

    return solution
