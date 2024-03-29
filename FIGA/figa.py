import copy
from math import exp, sqrt
from operator import attrgetter
from random import choice, sample, shuffle
from time import process_time
from typing import Deque, Dict, List, Tuple, Union

from numpy import ceil, random

from common import (check_are_identical,
                    check_iterations_termination_condition,
                    check_seconds_termination_condition, rand)
from constants import INT_MAX
from destination import Destination
from FIGA.figaSolution import FIGASolution
from FIGA.operators import (DBS_mutation, DBT_mutation, FBR_crossover,
                            FBS_mutation, LDHR_mutation, PBS_mutator,
                            SBCR_crossover, TWBLC_mutation, TWBR_mutation,
                            TWBS_mutation, VE_mutation)
from FIGA.parameters import (FBR_CROSSOVER_MAX_VEHICLES,
                             MAX_SIMULTANEOUS_MUTATIONS,
                             SBRC_CROSSOVER_MAX_VEHICLES,
                             TOURNAMENT_PROBABILITY_SELECT_BEST)
from problemInstance import ProblemInstance
from vehicle import Vehicle

# operators' statistics
initialiser_execution_time: float=0.0
feasible_initialisations: int=0
crossover_invocations: int=0
crossover_acceptances: Dict[int, int]={}
mutation_invocations: int=0
mutation_acceptances: Dict[int, int]={}
metropolis_returns: Dict[int, int]={1:0, 2:0, 3:0, 4:0}

def find_feasible_point(instance: ProblemInstance, solution: FIGASolution, new_destination: Destination) -> bool:
    longest_wait_time, longest_waiting_point = 0.0, (instance.amount_of_vehicles, 0)

    for v, vehicle in enumerate(solution.vehicles):
        if vehicle.current_capacity + new_destination.node.demand <= instance.capacity_of_vehicles:
            for d, destination in enumerate(vehicle.destinations[1:], 1):
                vehicle.destinations.insert(d, new_destination)

                if vehicle.calculate_destinations_time_windows(instance, start_from=d):
                    vehicle.current_capacity += new_destination.node.demand
                    return
                else:
                    vehicle.destinations.pop(d)
                    vehicle.calculate_destinations_time_windows(instance, start_from=d)

                    if destination.wait_time > longest_wait_time:
                        longest_wait_time, longest_waiting_point = destination.wait_time, (v, d)
    return longest_waiting_point

def DTWIH_III(instance: ProblemInstance, _id: int) -> FIGASolution:
    sorted_nodes = sorted(list(instance.nodes.values())[1:], key=lambda n: n.ready_time) # sort every available node (except the depot, hence [1:] slice) by their ready_time
    range_of_sorted_nodes = int(ceil((len(instance.nodes) - 1) / 10))
    solution = FIGASolution(_id=_id)

    while sorted_nodes:
        shuffled_nodes_buffer = sorted_nodes[:min(range_of_sorted_nodes, len(sorted_nodes))] # get nodes from 0 to range_of_sorted_nodes; once these nodes have been inserted, they will be deleted do the next iteration gets the next "range_of_sorted_nodes" nodes
        shuffle(shuffled_nodes_buffer)

        for node in shuffled_nodes_buffer:
            new_destination = Destination(node=node)
            alternative_point = find_feasible_point(instance, solution, new_destination)

            if alternative_point:
                if len(solution.vehicles) < instance.amount_of_vehicles:
                    solution.vehicles.append(Vehicle.create_route(instance, node))
                    solution.vehicles[-1].calculate_destinations_time_windows(instance)
                    solution.vehicles[-1].current_capacity = node.demand
                else:
                    v, d = alternative_point
                    solution.vehicles[v].destinations.insert(d, new_destination)
                    solution.vehicles[v].calculate_destinations_time_windows(instance, start_from=d)
        del sorted_nodes[:range_of_sorted_nodes] # remove the nodes that have been added from the sorted nodes to be added

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

    for s, solution in enumerate(nondominated_set[:-1]): # len - 1 because in the next loop, s + 1 will do the comparison of the last non-dominated solution; we never need s and s_aux to equal the same value as there's no point comparing identical solutions
        if s not in solutions_to_remove:
            for s_aux, solution_auxiliary in enumerate(nondominated_set[s + 1:], s + 1): # s + 1 to len will perform the comparisons that have not been carried out yet; any solutions between indexes 0 and s + 1 have already been compared to the solution at index s, and + 1 is so that solution s is not compared to s
                if s_aux not in solutions_to_remove:
                    # we need to check if both solutions dominate one another; s may not dominate s_aux, but s_aux may dominate s, and if neither dominate each other, then they still remain in the non-dominated set
                    if is_nondominated(solution, solution_auxiliary):
                        solutions_to_remove.add(s)
                        break
                    elif is_nondominated(solution_auxiliary, solution) \
                            or check_are_identical(solution, solution_auxiliary): # this "or" clause removes identical solutions, but it is not needed as "mo_metropolis" prevents duplicate solutions prior to this
                        solutions_to_remove.add(s_aux)

    if solutions_to_remove:
        i = 0
        for s in range(len(nondominated_set)):
            if s not in solutions_to_remove:
                nondominated_set[i] = nondominated_set[s] # shift every solution whose list index is not in solutions_to_remove
                i += 1
        if i != len(nondominated_set): # i will not equal the non-dominated set length if there are solutions to remove
            del nondominated_set[i:] # MMOEASA limits its non-dominated set to 20, so do the same here (this is optional)
            # return process_time() if subject_solution in nondominated_set else None

def attempt_time_window_based_reorder(instance: ProblemInstance, solution: FIGASolution) -> None:
    i = 0
    while i < len(solution.vehicles) and len(solution.vehicles) < instance.amount_of_vehicles:
        if solution.vehicles[i].get_num_of_customers_visited() >= 2:
            for j, destination in enumerate(solution.vehicles[i].get_customers_visited()[1:], 2):
                if destination.arrival_time > destination.node.due_date:
                    solution.vehicles.insert(i + 1, Vehicle.create_route(instance, solution.vehicles[i].destinations[j:-1]))
                    del solution.vehicles[i].destinations[j:-1]

                    solution.vehicles[i].calculate_vehicle_load()
                    solution.vehicles[i].calculate_length_of_route(instance)
                    solution.vehicles[i].calculate_destination_time_window(instance, j - 1, j)

                    solution.vehicles[i + 1].calculate_vehicle_load()
                    solution.vehicles[i + 1].calculate_length_of_route(instance)
                    solution.vehicles[i + 1].calculate_destinations_time_windows(instance)

                    break
        i += 1
    solution.objective_function(instance)

def selection_tournament(nondominated_set: List[FIGASolution], population: List[FIGASolution], exclude_solution: FIGASolution=None, is_min_vehicle_member: bool=False) -> FIGASolution:
    if exclude_solution:
        # if the non-dominated set isn't empty, and contains at least two solutions or one solution that is not "exclude_solution", then it is possible to use the non-dominated set
        if nondominated_set and rand(1, 100) < TOURNAMENT_PROBABILITY_SELECT_BEST:
            if is_min_vehicle_member:
                subject_list = list(filter(lambda s: s is not exclude_solution and s.num_vehicles != population[0].num_vehicles, nondominated_set))
            else:
                subject_list = list(filter(lambda s: s is not exclude_solution, nondominated_set))
            if subject_list:
                return choice(subject_list)
        return choice(list(filter(lambda s: s is not exclude_solution, population)))
    return choice(nondominated_set if nondominated_set and rand(1, 100) < TOURNAMENT_PROBABILITY_SELECT_BEST else population)

def try_crossover(instance: ProblemInstance, parent_one: FIGASolution, parent_two: FIGASolution, crossover_probability: int) -> Tuple[FIGASolution, Union[int, None]]:
    if rand(1, 100) < crossover_probability:
        # global crossover_invocations
        # crossover_invocations += 1

        crossover_solution = None
        crossover = rand(1, 2)

        match crossover:
            case 1:
                crossover_solution = FBR_crossover(instance, parent_one, sample(parent_two.vehicles, rand(1, min(FBR_CROSSOVER_MAX_VEHICLES, len(parent_two.vehicles) - 1))))
            case 2:
                crossover_solution = SBCR_crossover(instance, parent_one, sample(parent_two.vehicles, rand(1, min(SBRC_CROSSOVER_MAX_VEHICLES, len(parent_two.vehicles) - 1))))
        """case 1:
            crossover_solution = ES_crossover(instance, parent_one, sample(parent_two.vehicles, rand(1, min([ES_CROSSOVER_MAX_VEHICLES, len(parent_two.vehicles) - 1, instance.amount_of_vehicles - len(parent_one.vehicles)]))))
        case 2:
            crossover_solution = PM_crossover(instance, parent_one, parent_two)"""

        return crossover_solution, crossover
    return parent_one, None

def try_mutation(instance: ProblemInstance, solution: FIGASolution, mutation_probability: int) -> Tuple[FIGASolution, Union[int, None]]:
    if rand(1, 100) < mutation_probability:
        # global mutation_invocations
        # mutation_invocations += 1

        failed_mutators = set()
        mutated_solution, result = copy.deepcopy(solution), None # make a copy solution as we don't want to mutate the original; the functions below are given the object by reference in Python

        while result is None:
            mutator = rand(1, 7, exclude_values=failed_mutators)

            match mutator:
                case 1:
                    result = PBS_mutator(instance, mutated_solution) # Partition-based Swap Mutator
                case 2:
                    result = DBT_mutation(instance, mutated_solution) # Distance-based Transfer Mutator
                case 3:
                    result = DBS_mutation(instance, mutated_solution) # Distance-based Swap Mutator
                case 4:
                    result = LDHR_mutation(instance, mutated_solution) # Low Distance High Ready-time Mutator
                case 5:
                    result = VE_mutation(instance, mutated_solution) # Vehicle Elimination Mutator
                case 6:
                    result = FBS_mutation(instance, mutated_solution) # Feasibility-based Swap Mutator
                case 7:
                    result = TWBLC_mutation(instance, mutated_solution) # Time-Window-based Local Crossover Mutator
            """case 1:
                result = TWBS_mutation(instance, mutated_solution) # Time-Window-based Swap Mutator
            case 2:
                result = TWBR_mutation(instance, mutated_solution) # Time-Window-based Reorder Mutator
            case 11:
                mutated_solution = ATBR_mutation(instance, mutated_solution) # Arrival-Time Based Reorder Mutator"""

            if result is None:
                if len(failed_mutators) == 6:
                    break
                failed_mutators.add(mutator)
        return result, mutator
    return solution, None

def calculate_cooling(i: int, temperature_max: float, temperature_min: float, temperature_stop: float, population_size: int, termination_condition: int) -> float:
    # the calculate_cooling function simulates the genetic algorithm's iterations, from start to termination
    # over and over again until it finds the cooling rate that gets a solution's temperature to "temperature_stop" at the same time that the "termination_condition" is reached
    jump_temperatures = (temperature_max - temperature_min) / float(population_size - 1) if population_size > 1 else 0.0
    temperature_aux = temperature_max - float(i) * jump_temperatures
    error = float(INT_MAX)
    max_error = 0.005 * float(termination_condition)
    cooling_rate = 0.995
    auxiliary_iterations = 0.0

    while abs(error) > max_error and not auxiliary_iterations > termination_condition: # the original MMOEASA "Calculate_cooling" doesn't have the second condition, but mine (without it) gets an infinite loop (use the "print"s below to see)
        #print(abs(error), maxError, cooling_rate, auxiliary_iterations)
        temperature = temperature_aux
        auxiliary_iterations = 0.0

        while temperature > temperature_stop:
            temperature *= cooling_rate
            auxiliary_iterations += 1.0

        #print(termination_condition, auxiliary_iterations)
        error = float(termination_condition) - auxiliary_iterations
        cooling_rate = cooling_rate + (0.05 / float(termination_condition)) if error > 0.0 else cooling_rate - (0.05 / float(termination_condition))

    return cooling_rate

def euclidean_distance_dispersion(child: FIGASolution, parent: FIGASolution) -> float:
    x1, y1 = child.total_distance, child.num_vehicles
    x2, y2 = parent.total_distance, parent.num_vehicles
    return sqrt(((x2 - x1) / 2 * parent.total_distance * .9) ** 2 + ((y2 - y1) / 2 * (parent.num_vehicles - 1)) ** 2)

def mo_metropolis(parent: FIGASolution, child: FIGASolution, temperature: float, population: List[FIGASolution]=None) -> FIGASolution:
    # global metropolis_returns
    if not population:
        population = []
    duplicate = any(check_are_identical(child, solution) for solution in population)

    if is_nondominated(parent, child) and not duplicate:
        # metropolis_returns[1] += 1
        return child
    # elif is_nondominated(child, parent):
    #     # metropolis_returns[2] += 1
    #     return parent
    else:
        # d_df is a simulated deterioration (difference between the new and old solution) between the multi-objective variables
        # the Metropolis function accepts a solution based on this deterioration when neither the parent nor child dominate
        # the reason the deterioration needs to be simulated is that it cannot be calculated in a multi objective case; in a single-objective case, the deterioration would simply be "solution one's objective - solution two's objective"
        # if the deterioration is low, there is a better chance that the Metropolis function will accept the child solution
        d_df = euclidean_distance_dispersion(child, parent)
        # deterioration per-temperature-per-temperature simply incorporates the parent's Simulated Annealing temperature into the acceptance probability of MO_Metropolis
        # the new calculation in the "else" clause reduces the probability of accepting duplicate solutions from being recorded at a low temperature, and vice versa for high temperatures
        d_pt_pt = d_df / temperature ** 2
        d_exp = exp(-1.0 * d_pt_pt)

        if random.randint(INT_MAX) / INT_MAX < d_exp: # Metropolis acceptance criterion result is accepted based on probability
            # metropolis_returns[3] += 1
            return child
        # metropolis_returns[4] += 1
        return parent

def FIGA(instance: ProblemInstance, population_size: int, termination_condition: int, termination_type: str, crossover_probability: int, mutation_probability: int, temperature_max: float, temperature_min: float, temperature_stop: float, progress_indication_steps: Deque[float]) -> Tuple[List[FIGASolution], Dict[str, int]]:
    population: List[FIGASolution] = list()
    nondominated_set: List[FIGASolution] = list()

    global initialiser_execution_time, feasible_initialisations, mutation_acceptances, crossover_acceptances
    feasible_initialisations = 0
    start = process_time()
    for i in range(0, population_size):
        population.insert(i, DTWIH_III(instance, i))
        population[i].temperature = population[i].default_temperature = temperature_max - float(i) * ((temperature_max - temperature_min) / float(population_size - 1))
        population[i].cooling_rate = calculate_cooling(i, temperature_max, temperature_min, temperature_stop, population_size, termination_condition)
        if population[i].feasible:
            feasible_initialisations += 1
    initialiser_execution_time = round((process_time() - start) * 1000, 3)

    terminate = False
    iterations = 0
    last_nds_update = None
    while not terminate:
        if population[0].temperature <= temperature_stop:
            for s in range(population_size):
                population[s].temperature = population[s].default_temperature

            """if nondominated_set:
                new_generation = []
                nondominated_solution = copy.deepcopy(min(nondominated_set, key=attrgetter("num_vehicles")))
                nondominated_solution.default_temperature = population[0].default_temperature
                nondominated_solution.temperature = population[0].temperature * population[0].cooling_rate
                nondominated_solution.cooling_rate = population[0].cooling_rate
                new_generation.append(nondominated_solution)

                for s in range(1, population_size):
                    crossover_parent_two = choice(population)
                    solution = copy.deepcopy(choice(nondominated_set))
                    solution.default_temperature = population[s].default_temperature
                    solution.temperature = population[s].temperature
                    solution.cooling_rate = population[s].cooling_rate

                    if not crossover_parent_two.feasible and s == 1: # in this case, we may have selected an infeasible solution from the population, so try to correct it first
                        attempt_time_window_based_reorder(instance, crossover_parent_two)

                    child, crossover = try_crossover(instance, solution, crossover_parent_two, crossover_probability)
                    if crossover:
                        # check_nondominated_set_acceptance(nondominated_set, child)
                        nds_update = check_nondominated_set_acceptance(nondominated_set, child)
                        if nds_update:
                            last_nds_update = nds_update - start
                            nds_str = ""
                            for s_aux, solution in enumerate(nondominated_set):
                                nds_str += f"{solution.total_distance},{solution.num_vehicles}" + (" ||| " if s_aux < len(nondominated_set) - 1 else "")
                            print(nds_str)
                    # mutations = []
                    for _ in range(rand(1, MAX_SIMULTANEOUS_MUTATIONS)):
                        mutated_child, mutator = try_mutation(instance, mo_metropolis(solution, child, solution.temperature), 100)
                        if mutated_child is not None:
                            child = mutated_child
                        else:
                            break
                        if mutator:
                            # check_nondominated_set_acceptance(nondominated_set, child)
                            nds_update = check_nondominated_set_acceptance(nondominated_set, child)
                            if nds_update:
                                last_nds_update = nds_update - start
                                nds_str = ""
                                for s_aux, solution in enumerate(nondominated_set):
                                    nds_str += f"{solution.total_distance},{solution.num_vehicles}" + (" ||| " if s_aux < len(nondominated_set) - 1 else "")
                                print(nds_str)
                        # mutations.append(mutator)

                    #if not solution.feasible or mo_metropolis(instance, solution, child, solution.temperature, population=new_generation) is not solution:
                    new_generation.append(child)
                        # if crossover:
                        #     if not crossover in crossover_acceptances:
                        #         crossover_acceptances[crossover] = 1
                        #     else:
                        #         crossover_acceptances[crossover] += 1
                        # if mutations:
                        #     for mutator in filter(lambda m: m, mutations):
                        #         if not mutator in mutation_acceptances:
                        #             mutation_acceptances[mutator] = 1
                        #         else:
                        #             mutation_acceptances[mutator] += 1

                    new_generation[s].temperature *= new_generation[s].cooling_rate
                population = new_generation
                continue"""

        if nondominated_set:
            nondominated_solution = copy.deepcopy(min(nondominated_set, key=attrgetter("num_vehicles")))
            nondominated_solution.default_temperature = population[0].default_temperature
            nondominated_solution.temperature = population[0].temperature
            nondominated_solution.cooling_rate = population[0].cooling_rate
            population[0] = nondominated_solution

        for s, solution in enumerate(population):
            crossover_parent_two = selection_tournament(nondominated_set, population, exclude_solution=solution, is_min_vehicle_member=s == 0)

            if not solution.feasible:
                attempt_time_window_based_reorder(instance, solution)

            child, crossover = try_crossover(instance, solution, crossover_parent_two, crossover_probability)
            if crossover:
                check_nondominated_set_acceptance(nondominated_set, child)
                # nds_update = check_nondominated_set_acceptance(nondominated_set, child)
                # if nds_update:
                #     last_nds_update = nds_update - start
                #     nds_str = ""
                #     for s_aux, solution in enumerate(nondominated_set):
                #         nds_str += f"{solution.total_distance},{solution.num_vehicles}" + (" ||| " if s_aux < len(nondominated_set) - 1 else "")
                #     print(nds_str)
            # mutations = []
            for _ in range(rand(1, MAX_SIMULTANEOUS_MUTATIONS)):
                mutated_child, mutator = try_mutation(instance, mo_metropolis(solution, child, solution.temperature), mutation_probability)
                if mutated_child is not None:
                    child = mutated_child
                else:
                    break
                if mutator:
                    check_nondominated_set_acceptance(nondominated_set, child)
                    # nds_update = check_nondominated_set_acceptance(nondominated_set, child)
                    # if nds_update:
                    #     last_nds_update = nds_update - start
                    #     nds_str = ""
                    #     for s_aux, solution in enumerate(nondominated_set):
                    #         nds_str += f"{solution.total_distance},{solution.num_vehicles}" + (" ||| " if s_aux < len(nondominated_set) - 1 else "")
                    #     print(nds_str)
                # mutations.append(mutator)

            if not solution.feasible or mo_metropolis(solution, child, solution.temperature, population=population) is not solution:
                population[s] = child
                # if crossover:
                #     if not crossover in crossover_acceptances:
                #         crossover_acceptances[crossover] = 1
                #     else:
                #         crossover_acceptances[crossover] += 1
                # if mutations:
                #     for mutator in filter(lambda m: m, mutations):
                #         if not mutator in mutation_acceptances:
                #             mutation_acceptances[mutator] = 1
                #         else:
                #             mutation_acceptances[mutator] += 1

            population[s].temperature *= population[s].cooling_rate
        iterations += 1

        if termination_type == "iterations":
            terminate = check_iterations_termination_condition(iterations, termination_condition, len(nondominated_set), population, progress_indication_steps)
        elif termination_type == "seconds":
            terminate = check_seconds_termination_condition(start, termination_condition, len(nondominated_set), population, progress_indication_steps)

    global crossover_invocations, mutation_invocations, metropolis_returns
    statistics = {
        "iterations": iterations,
        "initialiser_execution_time": f"{initialiser_execution_time} milliseconds",
        "feasible_initialisations": feasible_initialisations,
        "crossover_invocations": crossover_invocations,
        "crossover_acceptances": dict(sorted(crossover_acceptances.items())),
        "total_accepted_crossovers": sum(crossover_acceptances.values()),
        "mutation_invocations": mutation_invocations,
        "mutation_acceptances": dict(sorted(mutation_acceptances.items())),
        "total_accepted_mutations": sum(mutation_acceptances.values()),
        "metropolis_returns": metropolis_returns,
        "final_nondominated_set_update": f"{round(last_nds_update, 1)}s" if last_nds_update else None
    }

    return nondominated_set, statistics
