import copy
from random import choice, sample, shuffle
from time import process_time
from typing import Deque, Dict, List, Tuple, Union

from numpy import round

from common import (check_are_identical,
                    check_iterations_termination_condition,
                    check_seconds_termination_condition, rand)
from constants import INT_MAX
from destination import Destination
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.auxiliaries import (get_nondominated_set, get_unique_set,
                                is_nondominated, mmoeasa_is_nondominated)
from Ombuki.constants import (GREEDY_PERCENT,
                              TOURNAMENT_PROBABILITY_SELECT_BEST,
                              TOURNAMENT_SET_SIZE)
from Ombuki.ombukiSolution import OmbukiSolution
from Ombuki.operators import crossover, mutation
from problemInstance import ProblemInstance
from vehicle import Vehicle

initialiser_execution_time: int=0
feasible_initialisations: int=0
crossover_invocations: int=0
crossover_successes: int=0
mutation_invocations: int=0
mutation_successes: int=0

def generate_random_solution(instance: ProblemInstance, _id: int) -> Union[OmbukiSolution, MMOEASASolution]:
    solution = OmbukiSolution(_id=_id) if instance.acceptance_criterion == "Ombuki" else MMOEASASolution(_id=_id)
    nodes = list(instance.nodes.values())[1:]
    shuffle(nodes)

    for node in nodes:
        feasible_vehicles = list(filter(lambda v: v.current_capacity + node.demand < instance.capacity_of_vehicles, solution.vehicles))
        vehicle = choice(feasible_vehicles) if feasible_vehicles else None
        
        if vehicle is None:
            solution.vehicles.append(Vehicle.create_route(instance, node=node))
            solution.vehicles[-1].current_capacity = node.demand
        else:
            vehicle.destinations.insert(len(vehicle.destinations) - 1, Destination(node=node))
            vehicle.current_capacity += node.demand
            """vehicle = rand(0, instance.amount_of_vehicles - 1, exclude_values=infeasible_vehicles) # generate a random number between 1 and max vehicles instead of 1 to current num of vehicles; to keep the original probability of inserting the node to a new vehicle

            if vehicle < len(solution.vehicles) and solution.vehicles[vehicle].current_capacity + instance.nodes[i].demand <= instance.capacity_of_vehicles:
                solution.vehicles[vehicle].destinations.insert(len(solution.vehicles[vehicle].destinations) - 1, Destination(node=instance.nodes[i]))
                solution.vehicles[vehicle].current_capacity += instance.nodes[i].demand
                inserted = True
            elif len(infeasible_vehicles) == len(solution.vehicles): # if every vehicle in the solution cannot occupy a new destination, then make a new vehicle
                solution.vehicles.append(Vehicle.create_route(instance, node=instance.nodes[i]))
                solution.vehicles[-1].current_capacity = instance.nodes[i].demand
                inserted = True
            else:
                infeasible_vehicles.add(vehicle)"""

    solution.calculate_length_of_routes(instance)
    solution.calculate_routes_time_windows(instance)
    solution.objective_function(instance)

    return solution

def generate_greedy_solution(instance: ProblemInstance, _id: int) -> Union[OmbukiSolution, MMOEASASolution]:
    solution = OmbukiSolution(_id=_id, vehicles=[Vehicle.create_route(instance)]) if instance.acceptance_criterion == "Ombuki" else MMOEASASolution(_id=_id, vehicles=[Vehicle.create_route(instance)])
    unvisited_nodes = list(range(1, len(instance.nodes)))
    vehicle = 0

    while unvisited_nodes:
        node = instance.nodes[choice(unvisited_nodes)]
        solution.vehicles[vehicle].destinations.insert(len(solution.vehicles[vehicle].destinations) - 1, Destination(node=node)) # first, insert the randomly selected vehicle to start a greedy search from
        solution.vehicles[vehicle].current_capacity = node.demand
        unvisited_nodes.remove(node.number) # the randomly chosen node was inserted, so it can be removed from the unvisited nodes

        while solution.vehicles[vehicle].current_capacity <= instance.capacity_of_vehicles:
            closest_node = None
            distance_of_closest = float(INT_MAX)

            for u_node in unvisited_nodes: # for every unvisited node, find the one closest to the node last inserted
                distance = instance.get_distance(node.number, u_node)
                if distance < distance_of_closest:
                    closest_node = u_node
                    distance_of_closest = distance

            if closest_node and not solution.vehicles[vehicle].current_capacity + instance.nodes[closest_node].demand > instance.capacity_of_vehicles:
                solution.vehicles[vehicle].destinations.insert(len(solution.vehicles[vehicle].destinations) - 1, Destination(node=instance.nodes[closest_node]))
                solution.vehicles[vehicle].current_capacity += instance.nodes[closest_node].demand
                node = solution.vehicles[vehicle].destinations[-2].node
                unvisited_nodes.remove(closest_node)
            else:
                if closest_node: # closest node will be given a value if there are still nodes to insert, so if the current vehicle cannot occupy the closest node due to capacity then create a new vehicle
                    solution.vehicles.append(Vehicle.create_route(instance))
                    vehicle += 1
                break

    solution.calculate_length_of_routes(instance)
    solution.calculate_routes_time_windows(instance)
    solution.objective_function(instance)

    return solution

def pareto_rank(instance: ProblemInstance, population: List[Union[OmbukiSolution, MMOEASASolution]]) -> int:
    curr_rank = 1
    unranked_solutions = population.copy()
    num_rank_ones = 0

    while unranked_solutions:
        nondominated_set = get_nondominated_set(unranked_solutions, mmoeasa_is_nondominated if instance.acceptance_criterion == "MMOEASA" else is_nondominated)
        if nondominated_set: # if there are no non-dominated solutions, then assign every remaining unranked solution the same rank (curr_rank)
            for solution in nondominated_set: # assign the current rank to every non-dominated solution in the population; curr_rank will equal 1 during the first iteration, then 2 in the next, and so on
                solution.rank = curr_rank
                if curr_rank == 1:
                    num_rank_ones += 1
            unranked_solutions = list(filter(lambda s: s not in nondominated_set, unranked_solutions)) # once solutions have been assigned a rank, remove them from the solutions to be ranked and start the Pareto-rank again with curr_rank + 1
        else:
            for solution in unranked_solutions:
                solution.rank = curr_rank
            if curr_rank == 1:
                num_rank_ones += len(unranked_solutions)
            break
        curr_rank += 1

    return num_rank_ones

def original_feasible_network_transformation(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution]) -> Union[OmbukiSolution, MMOEASASolution]:
    # create a new vehicle with the first destination of the first route as a starting point
    vehicles = [Vehicle.create_route(instance, solution.vehicles[0].destinations[1].node)]
    transformed_solution = OmbukiSolution(_id=solution.id, vehicles=vehicles) if instance.acceptance_criterion == "Ombuki" else MMOEASASolution(_id=solution.id, vehicles=vehicles)
    transformed_solution.vehicles[0].current_capacity += solution.vehicles[0].destinations[1].node.demand
    transformed_solution.vehicles[0].calculate_destination_time_window(instance, 0, 1)

    v = 0
    for vehicle in solution.vehicles:
        for destination in vehicle.get_customers_visited()[1 if vehicle is solution.vehicles[0] else 0:]: # if "vehicle" is the first vehicle, we can start the list at destination 2 as we already inserted the first one above 
            # try to rebuild the routes as they were, but...
            if transformed_solution.vehicles[v].current_capacity + destination.node.demand <= instance.capacity_of_vehicles and transformed_solution.vehicles[v].destinations[-2].departure_time + instance.get_distance(transformed_solution.vehicles[v].destinations[-2].node.number, destination.node.number) <= destination.node.due_date:
                transformed_solution.vehicles[v].destinations.insert(len(transformed_solution.vehicles[v].destinations) - 1, copy.deepcopy(destination))
                transformed_solution.vehicles[v].calculate_destination_time_window(instance, -3, -2)
                transformed_solution.vehicles[v].calculate_destination_time_window(instance, -2, -1)
            else: # ... every time an infeasible insertion is found (i.e. by inserting a destination to the end of the route being rebuilt, feasibility is broken), create a new vehicle and start rebuilding again from the new vehicle (v += 1 moves to the next vehicle)
                transformed_solution.vehicles.append(Vehicle.create_route(instance, destination.node))
                v += 1
                transformed_solution.vehicles[v].calculate_destinations_time_windows(instance)

            transformed_solution.vehicles[v].current_capacity += destination.node.demand

    transformed_solution.calculate_length_of_routes(instance)
    transformed_solution.objective_function(instance)

    return transformed_solution

def modified_feasible_network_transformation(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution]) -> Union[OmbukiSolution, MMOEASASolution]:
    # largely the same as the "original_feasible_network_transformation", except when a new vehicle cannot be created because of the problem instance's limit, the infeasible insertion is appended to the end of the route with the nearest final destination
    vehicles = [Vehicle.create_route(instance, solution.vehicles[0].destinations[1].node)]
    transformed_solution = OmbukiSolution(_id=solution.id, vehicles=vehicles) if instance.acceptance_criterion == "Ombuki" else MMOEASASolution(_id=solution.id, vehicles=vehicles)
    transformed_solution.vehicles[0].current_capacity += solution.vehicles[0].destinations[1].node.demand
    transformed_solution.vehicles[0].calculate_destination_time_window(instance, 0, 1)

    v = 0
    for vehicle in solution.vehicles:
        for destination in vehicle.get_customers_visited()[1 if vehicle is solution.vehicles[0] else 0:]:
            feasible_insertion = False
            vehicle_reset = False
            first_attempted_vehicle = v

            while not feasible_insertion:
                if transformed_solution.vehicles[v].current_capacity + destination.node.demand <= instance.capacity_of_vehicles and transformed_solution.vehicles[v].destinations[-2].departure_time + instance.get_distance(transformed_solution.vehicles[v].destinations[-2].node.number, destination.node.number) <= destination.node.due_date:
                    transformed_solution.vehicles[v].current_capacity += destination.node.demand
                    transformed_solution.vehicles[v].destinations.insert(len(transformed_solution.vehicles[v].destinations) - 1, copy.deepcopy(destination))
                    transformed_solution.vehicles[v].calculate_destination_time_window(instance, -3, -2)
                    transformed_solution.vehicles[v].calculate_destination_time_window(instance, -2, -1)
                    feasible_insertion = True
                elif v == instance.amount_of_vehicles - 1 or (vehicle_reset and v == first_attempted_vehicle): # if we've reached the end of the instance's permitted vehicle amount or the feasible insertion search has already been restarted once, and we've returned to the vehicle that the search started at
                    if vehicle_reset: # at this point, no feasible vehicle insertion was found, so select the best vehicle based on distance between the last destination and the destination to insert where capacity constraints are not violated; this solution is now infeasible
                        sorted_with_index = sorted(transformed_solution.vehicles, key=lambda veh: instance.get_distance(veh.destinations[-2].node.number, destination.node.number))
                        for infeasible_vehicle in sorted_with_index:
                            if infeasible_vehicle.current_capacity + destination.node.demand < instance.capacity_of_vehicles:
                                infeasible_vehicle.destinations.insert(infeasible_vehicle.get_num_of_customers_visited() + 1, copy.deepcopy(destination))
                                infeasible_vehicle.current_capacity += destination.node.demand
                                infeasible_vehicle.calculate_destination_time_window(instance, -3, -2)
                                infeasible_vehicle.calculate_destination_time_window(instance, -2, -1)
                                v = first_attempted_vehicle
                                break
                        break
                    else:
                        vehicle_reset = True # start again from vehicle zero as the destination being inserted may be feasible on the end of one of the previously built routes
                        v = 0
                else:
                    if len(transformed_solution.vehicles) < instance.amount_of_vehicles:
                        transformed_solution.vehicles.append(Vehicle.create_route(instance, destination.node))
                        transformed_solution.vehicles[-1].calculate_destinations_time_windows(instance)
                        transformed_solution.vehicles[-1].current_capacity += destination.node.demand
                        feasible_insertion = True
                    v += 1

    transformed_solution.calculate_length_of_routes(instance)
    transformed_solution.objective_function(instance)

    return transformed_solution

def relocate_final_destinations(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution]) -> Union[OmbukiSolution, MMOEASASolution]:
    relocated_solution = copy.deepcopy(solution)

    i = 0
    while i < len(relocated_solution.vehicles):
        feasible = relocated_solution.vehicles[i].calculate_destinations_time_windows(instance)
        j = i + 1 if i < len(relocated_solution.vehicles) - 1 else 0
        
        relocated_solution.vehicles[j].destinations.insert(1, relocated_solution.vehicles[i].destinations.pop(relocated_solution.vehicles[i].get_num_of_customers_visited())) # move a route's final destination to the following route's first destination
        relocated_solution.vehicles[i].calculate_length_of_route(instance)
        relocated_solution.vehicles[j].calculate_length_of_route(instance)
        relocated_solution.vehicles[j].current_capacity += relocated_solution.vehicles[j].destinations[1].node.demand

        if not relocated_solution.vehicles[j].current_capacity > instance.capacity_of_vehicles:
            relocated_solution.vehicles[j].calculate_destinations_time_windows(instance)
            swap_is_feasible = relocated_solution.vehicles[j].calculate_destinations_time_windows(instance)
            if feasible and not swap_is_feasible: # if, after relocation, the route's time windows are violated and the route before relocation was feasible, set "feasible" to false so that this chamge is reverted ...
                feasible = False
            else: # ... otherwise set it to whether the time windows were violated or not
                feasible = swap_is_feasible
        else:
            feasible = False

        if not feasible: # undo the previous relocation of a route's last destination to the next route's first, if it was not feasible
            relocated_solution.vehicles[i].destinations.insert(relocated_solution.vehicles[i].get_num_of_customers_visited() + 1, relocated_solution.vehicles[j].destinations.pop(1))
            relocated_solution.vehicles[i].calculate_length_of_route(instance)
            relocated_solution.vehicles[i].calculate_destination_time_window(instance, -3, -2)

            relocated_solution.vehicles[j].calculate_length_of_route(instance)
            relocated_solution.vehicles[j].current_capacity -= relocated_solution.vehicles[i].destinations[-2].node.demand
            relocated_solution.vehicles[j].calculate_destinations_time_windows(instance)

            i += 1
        else:
            relocated_solution.vehicles[i].current_capacity -= relocated_solution.vehicles[j].destinations[-2].node.demand
            relocated_solution.vehicles[i].calculate_destination_time_window(instance, -2, -1)

            if not relocated_solution.vehicles[i].get_num_of_customers_visited():
                del relocated_solution.vehicles[i]
            else:
                i += 1

    relocated_solution.objective_function(instance)
    return relocated_solution

def routing_scheme(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution], use_original: bool) -> Union[OmbukiSolution, MMOEASASolution]:
    attempt_feasible_network_transformation = original_feasible_network_transformation if use_original else modified_feasible_network_transformation
    transformed_solution = attempt_feasible_network_transformation(instance, solution)
    relocated_solution = relocate_final_destinations(instance, transformed_solution)

    # return the relocated solution if it dominates the transformed solution is dominated by it, otherwise return the transformed solution
    if instance.acceptance_criterion == "MMOEASA":
        return relocated_solution if not transformed_solution.feasible or (relocated_solution.total_distance < transformed_solution.total_distance and relocated_solution.cargo_unbalance < transformed_solution.cargo_unbalance) else transformed_solution
    return relocated_solution if not transformed_solution.feasible or (relocated_solution.total_distance < transformed_solution.total_distance and relocated_solution.num_vehicles < transformed_solution.num_vehicles) else transformed_solution

def selection_tournament(instance: ProblemInstance, population: List[Union[OmbukiSolution, MMOEASASolution]], exclude_solution: Union[OmbukiSolution, MMOEASASolution]=None) -> Union[OmbukiSolution, MMOEASASolution]:
    tournament_set = sample(list(filter(lambda s: s is not exclude_solution, population)) if exclude_solution else population, TOURNAMENT_SET_SIZE) # in the else instance, there are no feasible (and therefore, rank 1) solutions, so work with the entire population instead

    if rand(1, 100) < TOURNAMENT_PROBABILITY_SELECT_BEST: # probability of selecting the best solution in the tournament set to be returned ...
        best_solution = tournament_set[0]

        for solution in tournament_set[1:]: # find the non-dominated solution of the 4 chosen solutions, for it to be returned
            if instance.acceptance_criterion == "MMOEASA":
                if mmoeasa_is_nondominated(best_solution, solution):
                    best_solution = solution
            elif is_nondominated(best_solution, solution):
                best_solution = solution

        return best_solution
    return choice(tournament_set) # ... otherwise, return a random solution of the tournament set

def crossover_probability(instance: ProblemInstance, parent_one: Union[OmbukiSolution, MMOEASASolution], parent_two: Union[OmbukiSolution, MMOEASASolution], probability: int, use_original: bool) -> Tuple[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]]:
    parent_one_copy, parent_two_copy = copy.deepcopy(parent_one), copy.deepcopy(parent_two) # we always need to make a deep copy because we don't want multiple instances of the same solution in the new generation

    if rand(1, 100) < probability:
        # global crossover_invocations, crossover_successes
        # crossover_invocations += 1

        return crossover(instance, parent_one_copy, parent_two_copy, use_original)
    return parent_one_copy, parent_two_copy

def mutation_probability(instance: ProblemInstance, solution: Union[OmbukiSolution, MMOEASASolution], probability: int, pending_copy: bool) -> Union[OmbukiSolution, MMOEASASolution]:
    if rand(1, 100) < probability:
        # global mutation_invocations, mutation_successes
        # mutation_invocations += 1

        mutated_solution = mutation(instance, copy.deepcopy(solution))

        if instance.acceptance_criterion == "MMOEASA":
            if mmoeasa_is_nondominated(solution, mutated_solution):
                # mutation_successes += 1
                return mutated_solution
        elif is_nondominated(solution, mutated_solution):
            # mutation_successes += 1
            return mutated_solution
    return copy.deepcopy(solution) if pending_copy else solution # we always need to make a deep copy because we don't want multiple instances of the same solution in the new generation

def Ombuki(instance: ProblemInstance, population_size: int, termination_condition: int, termination_type: str, crossover: int, mutation: int, use_original_operators: bool, progress_indication_steps: Deque[float]) -> Tuple[List[Union[OmbukiSolution, MMOEASASolution]], Dict[str, int]]:
    population: List[Union[OmbukiSolution, MMOEASASolution]] = list()

    global initialiser_execution_time, feasible_initialisations
    feasible_initialisations = 0
    start = process_time()
    num_greedy_solutions = int(round(float(population_size * GREEDY_PERCENT))) # by default, "GREEDY_PERCENT" is 10%, so 10% of the population (30 as 300 * 0.1) will be greedy solutions ...
    for i in range(0, num_greedy_solutions):
        greedy_solution = generate_greedy_solution(instance, i)
        if greedy_solution.feasible:
            feasible_initialisations += 1
        population.insert(i, greedy_solution)
    for i in range(num_greedy_solutions, population_size): # ... the other 90% will be random generations
        random_solution = generate_random_solution(instance, i)
        if random_solution.feasible:
            feasible_initialisations += 1
        population.insert(i, random_solution)
    pareto_rank(instance, population)
    initialiser_execution_time = round((process_time() - start) * 1000, 3)

    terminate = False
    iterations = 0
    while not terminate:
        new_generation = []

        for _ in range(int(round(population_size / 2))):
            parent_one = selection_tournament(instance, population)
            parent_two = selection_tournament(instance, population, exclude_solution=parent_one)

            child_one, child_two = crossover_probability(instance, parent_one, parent_two, crossover, use_original_operators)

            mutated_solution_one = mutation_probability(instance, child_one, mutation, child_one is parent_one)
            mutated_solution_two = mutation_probability(instance, child_two, mutation, child_two is parent_two)

            # the routing scheme is likely destructive of good solutions and will have no effect on feasible solutions, so only execute it on infeasible solutions
            new_generation.append(mutated_solution_one if mutated_solution_one.feasible else routing_scheme(instance, mutated_solution_one, use_original_operators))
            new_generation.append(mutated_solution_two if mutated_solution_two.feasible else routing_scheme(instance, mutated_solution_two, use_original_operators))
        num_rank_ones = pareto_rank(instance, new_generation)

        if list(filter(lambda s: s.feasible, population)): # if the population contains any workable (feasible) solutions, then check if we should retain any of them in the new generation (via elitism), otherwise just use the new generation
            # don't filter out duplicate solutions because we can't assume that Ombuki's original algorithm does this, right?
            # nondominated_rank_ones = get_unique_set(list(filter(lambda s: s not in new_generation, get_nondominated_set(list(filter(lambda s: s.rank == 1, population + new_generation)), mmoeasa_is_nondominated if instance.acceptance_criterion == "MMOEASA" else is_nondominated))))#
            nondominated_rank_ones = list(filter(lambda s: s not in new_generation, get_nondominated_set(list(filter(lambda s: s.rank == 1, population + new_generation)), mmoeasa_is_nondominated if instance.acceptance_criterion == "MMOEASA" else is_nondominated)))
            new_generation[:len(nondominated_rank_ones)] = nondominated_rank_ones
        population = new_generation

        iterations += 1
        if termination_type == "iterations":
            terminate = check_iterations_termination_condition(iterations, termination_condition, num_rank_ones, population, progress_indication_steps)
        elif termination_type == "seconds":
            terminate = check_seconds_termination_condition(start, termination_condition, num_rank_ones, population, progress_indication_steps)

    global crossover_invocations, crossover_successes, mutation_invocations, mutation_successes
    statistics = {
        "iterations": iterations,
        "initialiser_execution_time": f"{initialiser_execution_time} milliseconds",
        "feasible_initialisations": feasible_initialisations,
        "crossover_invocations": crossover_invocations,
        "crossover_successes": crossover_successes,
        "mutation_invocations": mutation_invocations,
        "mutation_successes": mutation_successes
    }

    # because MMOEASA only returns a non-dominated set with a size equal to the population size, and Ombuki doesn't have a non-dominated set with a restricted size, the algorithm needs to select (unbiasedly) a fixed amount of rank 1 solutions for a fair evaluation
    return list(filter(lambda s: s.rank == 1 and s.feasible, get_unique_set(population))), statistics
