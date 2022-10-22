import copy
from random import choice
from time import process_time
from itertools import islice
from MMOEASA.auxiliaries import is_nondominated, ombuki_is_nondominated, check_nondominated_set_acceptance
from MMOEASA.operators import mutation1, mutation2, mutation3, mutation4, mutation5, mutation6, mutation7, mutation8, mutation9, mutation10, crossover1
from MMOEASA.constants import MAX_SIMULTANEOUS_MUTATIONS
from MMOEASA.mmoeasaSolution import MMOEASASolution
from Ombuki.ombukiSolution import OmbukiSolution
from problemInstance import ProblemInstance
from destination import Destination
from vehicle import Vehicle
from constants import INT_MAX
from common import rand, check_iterations_termination_condition, check_seconds_termination_condition
from typing import Callable, Deque, List, Tuple, Union, Dict
from numpy import sqrt, exp, random

initialiser_execution_time: int=0
feasible_initialisations: int=0
crossover_invocations: int=0
crossover_successes: int=0
mutation_invocations: int=0
mutation_successes: int=0

def TWIH(instance: ProblemInstance) -> Union[MMOEASASolution, OmbukiSolution]:
    sorted_nodes = sorted([value for _, value in islice(instance.nodes.items(), 1, len(instance.nodes))], key=lambda x: x.ready_time) # sort every customer (except the depot; "islice" starts the list from node 1) by their ready_time

    solution = MMOEASASolution(_id=0) if instance.acceptance_criterion == "MMOEASA" else OmbukiSolution(_id=0)
    s = 0

    for _ in range(0, instance.amount_of_vehicles - 1):
        if not s < len(instance.nodes) - 1: # end initialisation if the list of nodes iterator has allocated every node
            break
        vehicle = Vehicle.create_route(instance)

        while s < len(instance.nodes) - 1 and vehicle.current_capacity + sorted_nodes[s].demand < instance.capacity_of_vehicles:
            vehicle.destinations.insert(len(vehicle.destinations) - 1, Destination(node=sorted_nodes[s]))
            vehicle.current_capacity += float(sorted_nodes[s].demand)
            s += 1

        solution.vehicles.append(vehicle)

    solution.calculate_routes_time_windows(instance)
    solution.calculate_vehicles_loads()
    solution.calculate_length_of_routes(instance)
    solution.objective_function(instance)

    return solution

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

def crossover(instance: ProblemInstance, I: Union[MMOEASASolution, OmbukiSolution], population: List[Union[MMOEASASolution, OmbukiSolution]], P_crossover: int) -> Union[MMOEASASolution, OmbukiSolution]:
    if rand(1, 100) <= P_crossover:
        # global crossover_invocations
        # crossover_invocations += 1

        return crossover1(instance, copy.deepcopy(I), population)
    return I

def mutation(instance: ProblemInstance, I: Union[MMOEASASolution, OmbukiSolution], P_mutation: int, pending_copy: bool) -> Union[MMOEASASolution, OmbukiSolution]:
    if rand(1, 100) <= P_mutation:
        # global mutation_invocations
        # mutation_invocations += 1

        solution_copy = copy.deepcopy(I) if pending_copy else I

        match rand(1, 10):
            case 1:
                return mutation1(instance, solution_copy)
            case 2:
                return mutation2(instance, solution_copy)
            case 3:
                return mutation3(instance, solution_copy)
            case 4:
                return mutation4(instance, solution_copy)
            case 5:
                return mutation5(instance, solution_copy)
            case 6:
                return mutation6(instance, solution_copy)
            case 7:
                return mutation7(instance, solution_copy)
            case 8:
                return mutation8(instance, solution_copy)
            case 9:
                return mutation9(instance, solution_copy)
            case 10:
                return mutation10(instance, solution_copy)
    return I

def euclidean_distance_dispersion(instance: ProblemInstance, child: Union[MMOEASASolution, OmbukiSolution], parent: Union[MMOEASASolution, OmbukiSolution]) -> float:
    if instance.acceptance_criterion == "MMOEASA":
        # use "instance.Hypervolume_distance_unbalance" if you would like to use the distance imbalance as the second objective
        x1, y1 = child.total_distance, child.cargo_unbalance
        x2, y2 = parent.total_distance, parent.cargo_unbalance
        return sqrt(((x2 - x1) / 2 * instance.Hypervolume_total_distance) ** 2 + ((y2 - y1) / 2 * instance.Hypervolume_cargo_unbalance) ** 2)
    elif instance.acceptance_criterion == "OMBUKI":
        x1, y1 = child.total_distance, child.num_vehicles
        x2, y2 = parent.total_distance, parent.num_vehicles
        return sqrt(((x2 - x1) / 2 * instance.Hypervolume_total_distance) ** 2 + ((y2 - y1) / 2 * instance.Hypervolume_num_vehicles) ** 2)

def mo_metropolis(instance: ProblemInstance, parent: Union[MMOEASASolution, OmbukiSolution], child: Union[MMOEASASolution, OmbukiSolution], temperature: float, nondominated_check: Callable[[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]], bool]) -> MMOEASASolution:
    if nondominated_check(parent, child):
        return child
    elif temperature <= 0.00001:
        return parent
    else:
        # d_df is a simulated deterioration (difference between the new and old solution) between the multi-objective variables
        # the Metropolis function accepts a solution based on this deterioration when neither the parent nor child dominate
        # the reason the deterioration needs to be simulated is that it cannot be calculated in a multi objective case; in a single-objective case, the deterioration would simply be "solution one's objective - solution two's objective"
        # if the deterioration is low, there is a better chance that the Metropolis function will accept the child solution
        d_df = euclidean_distance_dispersion(instance, child, parent)
        # deterioration per-temperature-per-temperature simply incorporates the parent's Simulated Annealing temperature into the acceptance probability of MO_Metropolis
        d_pt_pt = d_df / temperature ** 2
        d_exp = exp(-1.0 * d_pt_pt) # Metropolis criterion

        if (random.randint(INT_MAX) / INT_MAX) < d_exp: # Metropolis acceptance criterion result is accepted based on probability
            return child
        else:
            return parent

def selection_tournament(population: List[Union[MMOEASASolution, OmbukiSolution]], nondominated_set: List[Union[MMOEASASolution, OmbukiSolution]], nondominated_check: Callable[[Union[OmbukiSolution, MMOEASASolution], Union[OmbukiSolution, MMOEASASolution]], bool], exclude_solution: Union[MMOEASASolution, OmbukiSolution]=None) -> Union[MMOEASASolution, OmbukiSolution]:
    population_members = list(filter(lambda s: s is not exclude_solution, population)) if exclude_solution else population
    feasible_population_members = list(filter(lambda s: s.feasible, population_members))
    solution_one = choice(feasible_population_members if feasible_population_members else population_members)
    nondominated_members = list(filter(lambda s: s is not exclude_solution and s is not solution_one, nondominated_set)) if nondominated_set else None
    solution_two = choice(nondominated_members) if nondominated_members else choice(population_members)
    return solution_one if nondominated_check(solution_two, solution_one) else solution_two

def MMOEASA(instance: ProblemInstance, population_size: int, multi_starts: int, termination_condition: int, termination_type: str, crossover_probability: int, mutation_probability: int, temperature_max: float, temperature_min: float, temperature_stop: float, progress_indication_steps: Deque[float]) -> Tuple[List[Union[OmbukiSolution, MMOEASASolution]], Dict[str, int]]:
    population: List[Union[MMOEASASolution, OmbukiSolution]] = list()
    nondominated_set: List[Union[MMOEASASolution, OmbukiSolution]] = list()

    global initialiser_execution_time, feasible_initialisations, crossover_successes, mutation_successes
    start = process_time()
    # the population is initialised with "population_size" amount of TWIH_solution copies
    TWIH_solution = TWIH(instance)
    if TWIH_solution.feasible:
        feasible_initialisations += 1
    for i in range(population_size):
        population.insert(i, copy.deepcopy(TWIH_solution))
        population[i].id = i
        population[i].default_temperature = temperature_max - float(i) * ((temperature_max - temperature_min) / float(population_size - 1))
        population[i].cooling_rate = calculate_cooling(i, temperature_max, temperature_min, temperature_stop, population_size, termination_condition)
    del TWIH_solution
    initialiser_execution_time = round((process_time() - start) * 1000, 3)

    terminate = False
    iterations = 0
    nondominated_check = is_nondominated if instance.acceptance_criterion == "MMOEASA" else ombuki_is_nondominated
    # the multi-start termination is commented out because it's used to calculate the number of iterations termination during the termination check
    # this is so multi-start doesn't terminate the algorithm when time is the termination condition
    while not terminate:
        for solution in population: # multi-start is used to restart the Simulated Annealing attributes of every solution
            solution.temperature = solution.default_temperature

        while population[0].temperature > temperature_stop and not terminate:
            # parent_two = selection_tournament(population, nondominated_set, nondominated_check)

            for s, solution in enumerate(population):
                child_solution = crossover(instance, solution, choice(list(filter(lambda s: s is not solution, population))), crossover_probability)# parent_two if parent_two is not solution else selection_tournament(population, nondominated_set, nondominated_check, exclude_solution=solution), crossover_probability)
                crossover_occurred = child_solution is not solution # if the copy is equal to the original solution, this means that no copy happened and, therefore, crossover did not occur
                # mutations = 0
                # for _ in range(0, rand(1, MAX_SIMULTANEOUS_MUTATIONS)): # MMOEASA can perform up to three mutations in a single generation
                #     solution_copy, mutation_occurred = mutation(instance, mo_metropolis(instance, solution, solution_copy, solution.temperature, nondominated_check), mutation_probability, solution_copy is solution)
                #     if mutation_occurred:
                #         mutations += 1
                mutated_solution = mutation(instance, mo_metropolis(instance, solution, child_solution, solution.temperature, nondominated_check), mutation_probability, crossover_occurred)
                mutation_occurred = mutated_solution is not child_solution and mutated_solution is not solution

                if crossover_occurred or mutation_occurred:
                    population[s] = mo_metropolis(instance, solution, mutated_solution, solution.temperature, nondominated_check)
                    # if the metropolis function chose to overwrite the parent and the child is feasible and the child was added to the non-dominated set
                    if population[s] is not solution and population[s].feasible:
                        check_nondominated_set_acceptance(nondominated_set, population[s], nondominated_check)
                        """if check_nondominated_set_acceptance(nondominated_set, population[s], nondominated_check):
                            if crossover_occurred:
                                crossover_successes += 1
                            if mutation_occurred:
                                mutation_successes += 1"""
                        # if mutations > 0:
                        #     mutation_successes += mutations

                """# uncomment this code if you'd like a solution to be written to a CSV
                if instance.acceptance_criterion == "MMOEASA" and nondominated_set:
                    MMOEASA_write_solution_for_validation(nondominated_set[0], instance.capacity_of_vehicles)"""

                population[s].temperature *= population[s].cooling_rate
            iterations += 1

            if termination_type == "iterations":
                terminate = check_iterations_termination_condition(iterations, termination_condition * multi_starts, len(nondominated_set), population, progress_indication_steps)
            elif termination_type == "seconds":
                terminate = check_seconds_termination_condition(start, termination_condition, len(nondominated_set), population, progress_indication_steps)

    global crossover_invocations, mutation_invocations
    statistics = {
        "initialiser_execution_time": f"{initialiser_execution_time} milliseconds",
        "feasible_initialisations": feasible_initialisations,
        "crossover_invocations": crossover_invocations,
        "crossover_successes": crossover_successes,
        "mutation_invocations": mutation_invocations,
        "mutation_successes": mutation_successes
    }

    return nondominated_set, statistics
