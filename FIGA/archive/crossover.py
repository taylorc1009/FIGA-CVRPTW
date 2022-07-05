import copy
import time
from threading import Lock
from multiprocessing import Process
from typing import Dict, Set, Tuple
from FIGA.figaSolution import FIGASolution
from destination import Destination
from problemInstance import ProblemInstance
from common import rand, INT_MAX

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

    crossover_solution.calculate_nodes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads(instance)

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

def crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution) -> FIGASolution:
    crossover_solution, nodes_to_insert, parent_two_destinations = initialise_decision_tree_prerequisites(instance, parent_one, parent_two)

    stats_record = {destination.node.number: CrossoverPositionStats() for destination in parent_two_destinations}
    start = time.time()
    crossover_solution_copy = copy.deepcopy(crossover_solution)
    crossover_solution = crossover_evaluation(instance, crossover_solution_copy, nodes_to_insert, copy.deepcopy(stats_record), stats_record, len(nodes_to_insert))
    copy_is = crossover_solution_copy is crossover_solution

    print(f"{round(time.time() - start, 1)}s", copy_is)
    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.calculate_nodes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads(instance)
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

def crossover_multithreaded(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution) -> FIGASolution:
    crossover_solution, nodes_to_insert, parent_two_destinations = initialise_decision_tree_prerequisites(instance, parent_one, parent_two)

    stats_record = {destination.node.number: CrossoverPositionStats() for destination in parent_two_destinations}
    start = time.time()
    crossover_solution_copy = copy.deepcopy(crossover_solution)
    result = {0: None}
    crossover_solution = crossover_evaluation_multithreaded(instance, crossover_solution_copy, nodes_to_insert, copy.deepcopy(stats_record), stats_record, len(nodes_to_insert), result)

    print(f"{round(time.time() - start, 1)}s", crossover_solution_copy is crossover_solution)

    crossover_solution.calculate_length_of_routes(instance)
    crossover_solution.calculate_nodes_time_windows(instance)
    crossover_solution.calculate_vehicles_loads(instance)
    crossover_solution.objective_function(instance)

    return crossover_solution

""" ^^^ Parallel Decision-tree-based Crossover Operator END ^^^ """
