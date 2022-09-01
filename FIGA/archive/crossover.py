import copy
from random import sample, shuffle
import time
from threading import Lock
from multiprocessing import Process
from typing import Dict, List, Set, Tuple
from FIGA.figaSolution import FIGASolution
from destination import Destination
from problemInstance import ProblemInstance
from constants import INT_MAX
from common import rand
from threading import Thread, currentThread
#from FIGA.figa import is_nondominated # if you move the double BCRC from this archive, this import will cause a circular import
from vehicle import Vehicle

class CrossoverPositionStats:
    def __init__(self) -> None:
        self.distance_from_previous = float(INT_MAX)
        self.distance_to_next = float(INT_MAX)

    def update_record(self, distance_from_previous: float, distance_to_next: float) -> None:
        self.distance_from_previous = float(distance_from_previous)
        self.distance_to_next = float(distance_to_next)

def initialise_decision_tree_prerequisites(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution) -> Tuple[FIGASolution, Set, List[Destination]]:
    crossover_solution = copy.deepcopy(parent_one)
    parent_two_destinations = parent_two.vehicles[rand(0, len(parent_two.vehicles) - 1)].get_customers_visited()
    nodes_to_remove = set([d.node.number for d in parent_two_destinations])
    nodes_to_insert = copy.deepcopy(nodes_to_remove)

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
                    break  # break, otherwise the while loop will start searching the next vehicle with "j" as the same value; without incrementing "i" and starting "j" at 0
            else:
                j += 1
        if increment:
            i += 1

    crossover_solution.calculate_routes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads()

    return crossover_solution, nodes_to_insert, parent_two_destinations

""" VVV Serial Decision-tree-based Crossover Operator START VVV """

def crossover_evaluation(instance: ProblemInstance, crossover_solution: FIGASolution, nodes_to_insert: Set[int], stats_record: Dict[int, CrossoverPositionStats], best_stats_record: Dict[int, CrossoverPositionStats], iteration: int) -> FIGASolution:
    if not iteration:
        for key, value in stats_record.items():
            best_stats_record[key].update_record(value.distance_from_previous, value.distance_to_next)
        return crossover_solution
    crossover_solution_final = None

    for node in list(nodes_to_insert):
        shortest_from_previous, shortest_to_next = best_stats_record[node].distance_from_previous, best_stats_record[node].distance_to_next

        for v, vehicle in enumerate(crossover_solution.vehicles):
            if vehicle.current_capacity + instance.nodes[node].demand <= instance.capacity_of_vehicles:
                for d, destination in enumerate(vehicle.destinations[1:], 1):
                    distance_from_previous = instance.get_distance(vehicle.destinations[d - 1].node.number, node)
                    distance_to_next = instance.get_distance(node, destination.node.number)

                    if vehicle.is_feasible_route(instance, additional_node=instance.nodes[node], position_of_additional=d) \
                    and ((distance_from_previous < shortest_from_previous and distance_to_next <= shortest_to_next)
                    or (distance_from_previous <= shortest_from_previous and distance_to_next < shortest_to_next)):

                        crossover_solution_copy = copy.deepcopy(crossover_solution)
                        crossover_solution_copy.vehicles[v].destinations.insert(d, Destination(node=instance.nodes[node]))
                        for d_aux in range(d, crossover_solution_copy.vehicles[v].get_num_of_customers_visited() + 1):
                            crossover_solution_copy.vehicles[v].calculate_destination_time_window(instance, d_aux - 1, d_aux)
                        crossover_solution_copy.vehicles[v].current_capacity += instance.nodes[node].demand
                        stats_record[node].update_record(distance_from_previous, distance_to_next)

                        evaluation_result = crossover_evaluation(instance, crossover_solution_copy, nodes_to_insert.difference({node}), stats_record, best_stats_record, iteration - 1)
                        if evaluation_result:
                            crossover_solution_final = evaluation_result
                            shortest_from_previous, shortest_to_next = distance_from_previous, shortest_to_next

    return crossover_solution_final

def crossover_tree(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution) -> FIGASolution:
    crossover_solution, nodes_to_insert, parent_two_destinations = initialise_decision_tree_prerequisites(instance, parent_one, parent_two)

    stats_record = {destination.node.number: CrossoverPositionStats() for destination in parent_two_destinations}
    start = time.time()
    crossover_solution_copy = copy.deepcopy(crossover_solution)
    crossover_solution = crossover_evaluation(instance, crossover_solution_copy, nodes_to_insert, copy.deepcopy(stats_record), stats_record, len(nodes_to_insert))
    copy_is = crossover_solution_copy is crossover_solution

    print(f"{round(time.time() - start, 1)}s", copy_is)
    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.calculate_routes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads()
    crossover_solution.objective_function(instance)

    return crossover_solution

""" ^^^ Serial Decision-tree-based Crossover Operator END ^^^ """

""" VVV Parallel Decision-tree-based Crossover Operator START VVV """

mutex = Lock()

def crossover_evaluation_multithreaded(instance: ProblemInstance, crossover_solution: FIGASolution, nodes_to_insert: Set[int], stats_record: Dict[int, CrossoverPositionStats], best_stats_record: Dict[int, CrossoverPositionStats], iteration: int, result: Dict[int, FIGASolution]) -> None:
    if not iteration:
        with mutex:
            for key, value in stats_record.items():
                best_stats_record[key].update_record(value.distance_from_previous, value.distance_to_next)
            result[0] = crossover_solution
    thread_pool = list()

    for node in list(nodes_to_insert):
        for v, vehicle in enumerate(crossover_solution.vehicles):
            if vehicle.current_capacity + instance.nodes[node].demand <= instance.capacity_of_vehicles:
                for d, destination in enumerate(vehicle.destinations[1:], 1):
                    if mutex.locked():
                        mutex.acquire()
                        mutex.release()
                    if ((instance.get_distance(vehicle.destinations[d - 1].node.number, node) < best_stats_record[node].distance_from_previous and instance.get_distance(node, destination.node.number) <= best_stats_record[node].distance_to_next)
                    or (instance.get_distance(vehicle.destinations[d - 1].node.number, node) <= best_stats_record[node].distance_from_previous and instance.get_distance(node, destination.node.number) < best_stats_record[node].distance_to_next)) \
                    and vehicle.is_feasible_route(instance, additional_node=instance.nodes[node], position_of_additional=d):

                        crossover_solution_copy = copy.deepcopy(crossover_solution)
                        crossover_solution_copy.vehicles[v].destinations.insert(d, Destination(node=instance.nodes[node]))
                        for d_aux in range(d, crossover_solution_copy.vehicles[v].get_num_of_customers_visited() + 1):
                            crossover_solution_copy.vehicles[v].calculate_destination_time_window(instance, d_aux - 1, d_aux)
                        crossover_solution_copy.vehicles[v].current_capacity += instance.nodes[node].demand
                        stats_record[node].update_record(instance.get_distance(vehicle.destinations[d - 1].node.number, node), instance.get_distance(node, destination.node.number))

                        t = Process(target=crossover_evaluation_multithreaded, args=(instance, crossover_solution_copy, nodes_to_insert.difference({node}), stats_record, best_stats_record, iteration - 1, result))
                        t.start()
                        thread_pool.append(t)
    for t in thread_pool:
        t.join()

def crossover_tree_multithreaded(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution) -> FIGASolution:
    crossover_solution, nodes_to_insert, parent_two_destinations = initialise_decision_tree_prerequisites(instance, parent_one, parent_two)

    stats_record = {destination.node.number: CrossoverPositionStats() for destination in parent_two_destinations}
    start = time.time()
    crossover_solution_copy = copy.deepcopy(crossover_solution)
    result = {0: None}
    crossover_solution = crossover_evaluation_multithreaded(instance, crossover_solution_copy, nodes_to_insert, copy.deepcopy(stats_record), stats_record, len(nodes_to_insert), result)

    print(f"{round(time.time() - start, 1)}s", crossover_solution_copy is crossover_solution)

    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.calculate_routes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads()
    crossover_solution.objective_function(instance)

    return crossover_solution

""" ^^^ Parallel Decision-tree-based Crossover Operator END ^^^ """

""" VVV Parallel double Best Cost Route Crossover START """

def set_up_crossover_child(instance: ProblemInstance, parent_one: FIGASolution, parent_two_destinations: List[Vehicle]) -> FIGASolution:
    child_solution = copy.deepcopy(parent_one)

    nodes_to_remove = set({d.node.number for d in parent_two_destinations}) # create a set containing the numbers of every node in parent_two_vehicle to be merged into parent_one's routes
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
    #child_solution.calculate_length_of_routes(instance) # this is not required here as the crossovers don't do any work with the total length of each route at this stage

    return child_solution

def crossover_thread(instance: ProblemInstance, primary_parent: FIGASolution, parent_two_vehicles: Vehicle, results: Dict[str, FIGASolution]) -> None:
    randomized_destinations = [destination for vehicle in parent_two_vehicles for destination in vehicle.get_customers_visited()]
    crossover_solution = set_up_crossover_child(instance, primary_parent, randomized_destinations)
    shuffle(randomized_destinations)

    for parent_destination in randomized_destinations:
        best_vehicle, best_position = (None,) * 2
        shortest_from_previous, shortest_to_next = (float(INT_MAX),) * 2
        highest_wait_time = 0.0
        lowest_ready_time_difference = float(INT_MAX)
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
                    elif not found_feasible_location:
                        ready_time_difference = abs(vehicle.destinations[j].node.ready_time - (crossover_solution.vehicles[i].destinations[j - 1].departure_time + distance_from_previous))
                        if crossover_solution.vehicles[i].destinations[j].wait_time > highest_wait_time and ready_time_difference < lowest_ready_time_difference:
                        # if no feasible insertion point has been found yet and the wait time of the previous destination is the highest that's been found, then record this as the best position
                            best_vehicle, best_position, highest_wait_time = i, j, crossover_solution.vehicles[i].destinations[j].wait_time

        if not found_feasible_location and len(crossover_solution.vehicles) < instance.amount_of_vehicles and best_vehicle is None:
            best_vehicle = len(crossover_solution.vehicles)
            crossover_solution.vehicles.append(Vehicle.create_route(instance, parent_destination)) # we don't need to give "Vehicle.create_route" a deep copy of the destination as it constructs an new Destination instance
        else:
            # best_vehicle and best_position will equal the insertion position before the vehicle with the longest wait time
            # that is if no feasible insertion point was found, otherwise it will equal the fittest feasible insertion point
            crossover_solution.vehicles[best_vehicle].destinations.insert(best_position, copy.deepcopy(parent_destination))

        crossover_solution.vehicles[best_vehicle].calculate_vehicle_load()
        crossover_solution.vehicles[best_vehicle].calculate_destinations_time_windows(instance)
        crossover_solution.vehicles[best_vehicle].calculate_length_of_route(instance)

    crossover_solution.objective_function(instance)
    results[currentThread().getName()] = crossover_solution

def crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two: List[Vehicle]) -> Tuple[FIGASolution, FIGASolution]:
    parent_one_vehicles = sample(parent_one.vehicles, rand(1, 3))
    parent_two_vehicles = sample(parent_two.vehicles, rand(1, 3))

    thread_results: Dict[str, FIGASolution] = {"child_one": None, "child_two": None}
    child_one_thread = Thread(name="child_one", target=crossover_thread, args=(instance, parent_one, parent_two_vehicles, thread_results))
    child_two_thread = Thread(name="child_two", target=crossover_thread, args=(instance, parent_two, parent_one_vehicles, thread_results))
    child_one_thread.start()
    child_two_thread.start()
    child_one_thread.join()
    child_two_thread.join()

    return thread_results["child_one"], thread_results["child_two"]

""" ^^^ Parallel double Best Cost Route Crossover END """
