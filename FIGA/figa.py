import copy
import time
import random
from typing import List, Dict, Tuple
from common import rand, check_iterations_termination_condition, check_seconds_termination_condition
from random import shuffle
from destination import Destination
from problemInstance import ProblemInstance
from FIGA.figaSolution import FIGASolution
from FIGA.operators import crossover, TWBS_mutation, TWBSw_mutation, WTBS_mutation, SWTBS_mutation, DBS_mutation, SDBS_mutation, TWBMF_mutation, TWBPB_mutation
from FIGA.constants import TOURNAMENT_PROBABILITY_SELECT_BEST
from vehicle import Vehicle
from numpy import ceil, random

# operators' statistics
initialiser_execution_time: int=0
feasible_initialisations: int=0
crossover_invocations: int=0
crossover_successes: int=0
mutation_invocations: int=0
mutation_successes: int=0

def DTWIH(instance: ProblemInstance) -> FIGASolution:
    sorted_nodes = sorted([node for _, node in instance.nodes.items() if node.number], key=lambda n: n.ready_time) # sort every available node by their ready_time
    num_routes = int(ceil(instance.amount_of_vehicles / 2))
    solution = FIGASolution(_id=0, vehicles=[Vehicle.create_route(instance) for _ in range(0, num_routes)])
    additional_vehicles = 0

    while sorted_nodes:
        range_of_sorted_nodes = num_routes if num_routes < len(sorted_nodes) else len(sorted_nodes) # if there are less remaining nodes than there are routes, set the range end to the number of remaining nodes
        nodes_to_insert = sorted_nodes[:range_of_sorted_nodes] # get nodes from 0 to range_of_sorted_nodes; once these nodes have been inserted, they will be deleted do the next iteration gets the next "range_of_sorted_nodes" nodes
        shuffle(nodes_to_insert)
        for i in range(range_of_sorted_nodes):
            index = i # index is used to keep track of the route that a node was inserted into so that the addition of that node's cargo demand can be added to that route
            if solution.vehicles[index].current_capacity + nodes_to_insert[i].demand <= instance.capacity_of_vehicles:
                solution.vehicles[index].destinations.insert(len(solution.vehicles[index].destinations) - 1, Destination(node=nodes_to_insert[i]))
            else: # a new vehicle had to be created in this instance because the vehicle to be inserted to couldn't occupy it due to capacity constraints
                # TODO: try picking the best location in an existing vehicle?
                index = (num_routes - 1) + additional_vehicles # set index to the number of expected routes plus the number of additional vehicles whose capacity is also full
                if solution.vehicles[index].current_capacity + nodes_to_insert[i].demand > instance.capacity_of_vehicles:
                    solution.vehicles.append(Vehicle.create_route(instance, nodes_to_insert[i]))
                    additional_vehicles += 1
                    index += 1
                else:
                    solution.vehicles[index].destinations.insert(len(solution.vehicles[index].destinations) - 1, Destination(node=nodes_to_insert[i]))
            solution.vehicles[index].current_capacity += nodes_to_insert[i].demand
        del sorted_nodes[:range_of_sorted_nodes] # remove the nodes that have been added from the sorted nodes to be added

    solution.calculate_nodes_time_windows(instance)
    solution.calculate_length_of_routes(instance)
    solution.objective_function(instance)

    return solution

def is_nondominated(old_solution: FIGASolution, new_solution: FIGASolution) -> bool:
    return (new_solution.total_distance < old_solution.total_distance and new_solution.num_vehicles <= old_solution.num_vehicles) or (new_solution.total_distance <= old_solution.total_distance and new_solution.num_vehicles < old_solution.num_vehicles)

def check_nondominated_set_acceptance(nondominated_set: List[FIGASolution], subject_solution: FIGASolution) -> None:
    if not subject_solution.feasible:
        return

    nondominated_set.append(subject_solution) # append the new solution to the non-dominated set; it will either remain or be removed by this procedure, depending on whether it dominates or not
    solutions_to_remove = set()

    if len(nondominated_set) > 1:
        for s, solution in enumerate(nondominated_set[:len(nondominated_set) - 1]): # len - 1 because in the next loop, s + 1 will do the comparison of the last non-dominated solution; we never need s and s_aux to equal the same value as there's no point comparing identical solutions
            for s_aux, solution_auxiliary in enumerate(nondominated_set[s + 1:], s + 1): # s + 1 to len will perform the comparisons that have not been carried out yet; any solutions between indexes 0 and s + 1 have already been compared to the solution at index s, and + 1 is so that solution s is not compared to s
                # we need to check if both solutions dominate one another; s may not dominate s_aux, but s_aux may dominate s, and if neither dominate each other, then they still remain in the non-dominated set
                if is_nondominated(solution, solution_auxiliary):
                    solutions_to_remove.add(s)
                elif is_nondominated(solution_auxiliary, solution) \
                        or (solution.total_distance == solution_auxiliary.total_distance and solution.num_vehicles == solution_auxiliary.num_vehicles): # this "or" clause removes identical solutions
                    solutions_to_remove.add(s_aux)

        if solutions_to_remove:
            i = 0
            for s in range(len(nondominated_set)):
                if s not in solutions_to_remove:
                    nondominated_set[i] = nondominated_set[s] # shift every solution whose list index is not in solutions_to_remove
                    i += 1
            if i != len(nondominated_set): # i will not equal the non-dominated set length if there are solutions to remove
                if i > 20:
                    i = 20 # MMOEASA limits its non-dominated set to 20, so do the same here (this is optional)
                del nondominated_set[i:]

def selection_tournament(nondominated_set: List[FIGASolution], population: List[FIGASolution]) -> FIGASolution:
    return random.choice(nondominated_set if nondominated_set and rand(1, 100) < TOURNAMENT_PROBABILITY_SELECT_BEST else population)

def try_crossover(instance, parent_one: FIGASolution, parent_two: FIGASolution, crossover_probability) -> FIGASolution:
    if rand(1, 100) < crossover_probability:
        global crossover_invocations, crossover_successes
        crossover_invocations += 1

        crossover_solution = crossover(instance, parent_one, parent_two.vehicles[rand(0, len(parent_two.vehicles) - 1)])

        if is_nondominated(parent_one, crossover_solution):
            crossover_successes += 1
        return crossover_solution
    return parent_one

def try_mutation(instance, solution: FIGASolution, mutation_probability: int) -> FIGASolution:
    if rand(1, 100) < mutation_probability:
        global mutation_invocations, mutation_successes
        mutation_invocations += 1

        mutated_solution = copy.deepcopy(solution) # make a copy solution as we don't want to mutate the original; the functions below are given the object by reference in Python
        probability = rand(1, 8)

        if probability == 1:
            mutated_solution = TWBS_mutation(instance, mutated_solution) # Time-Window-based Sort Mutator
        elif probability == 2:
            mutated_solution = TWBSw_mutation(instance, mutated_solution) # Time-Window-based Swap Mutator
        elif probability == 3:
            mutated_solution = WTBS_mutation(instance, mutated_solution) # Wait-Time-based Swap Mutator
        elif probability == 4:
            mutated_solution = SWTBS_mutation(instance, mutated_solution) # Single Wait-Time-based Swap Mutator
        elif probability == 5:
            mutated_solution = DBS_mutation(instance, mutated_solution) # Distance-based Swap Mutator
        elif probability == 6:
            mutated_solution = SDBS_mutation(instance, mutated_solution) # Single Distance-based Swap Mutator
        elif probability == 7:
            mutated_solution = TWBMF_mutation(instance, mutated_solution) # Time-Window-based Move Forward Mutator
        elif probability == 8:
            mutated_solution = TWBPB_mutation(instance, mutated_solution) # Time-Window-based Push-back Mutator

        if is_nondominated(solution, mutated_solution):
            mutation_successes += 1
            return mutated_solution
    return solution

def FIGA(instance: ProblemInstance, population_size: int, termination_condition: int, termination_type: str, crossover_probability: int, mutation_probability: int) -> Tuple[List[FIGASolution], Dict[str, int]]:
    population: List[FIGASolution] = list()
    nondominated_set: List[FIGASolution] = list()

    global initialiser_execution_time, feasible_initialisations
    initialiser_execution_time = time.time()
    for i in range(0, population_size):
        population.insert(i, DTWIH(instance))
        population[i].id = i
        if population[i].feasible:
            feasible_initialisations += 1
    initialiser_execution_time = round((time.time() - initialiser_execution_time) * 1000, 3)

    start = time.time()
    terminate = False
    iterations = 0
    while not terminate:
        crossover_parent_two = selection_tournament(nondominated_set, population)
        for s, solution in enumerate(population):
            child = try_crossover(instance, solution, crossover_parent_two, crossover_probability)
            child = try_mutation(instance, child, mutation_probability)

            if not solution.feasible or is_nondominated(solution, child):
                population[s] = child
                check_nondominated_set_acceptance(nondominated_set, population[s]) # this procedure will add the dominating child to the non-dominated set for us, if it should be there
        iterations += 1

        if termination_type == "iterations":
            terminate = check_iterations_termination_condition(iterations, termination_condition, len(nondominated_set))
        elif termination_type == "seconds":
            terminate = check_seconds_termination_condition(start, termination_condition, len(nondominated_set))

    global crossover_invocations, crossover_successes, mutation_invocations, mutation_successes
    statistics = {
        "initialiser_execution_time": f"{initialiser_execution_time} milliseconds",
        "feasible_initialisations": feasible_initialisations,
        "crossover_invocations": crossover_invocations,
        "crossover_successes": crossover_successes,
        "mutation_invocations": mutation_invocations,
        "mutation_successes": mutation_successes
    }

    return nondominated_set, statistics
