from common import rand
from FIGA.figaSolution import FIGASolution
from FIGA.operators import select_random_vehicle, swap
from problemInstance import ProblemInstance


def MMOEASA_mutation5(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution:
    # select two random vehicles and random destinations from each of them, then swap the two selections
    random_origin_vehicle = select_random_vehicle(solution, customers_required=1)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    random_destination_vehicle = select_random_vehicle(solution, customers_required=1, exclude_values=set((random_origin_vehicle,)))
    destination_position = rand(1, solution.vehicles[random_destination_vehicle].get_num_of_customers_visited())

    swap(solution.vehicles[random_origin_vehicle].destinations, origin_position, destination_position, l2=solution.vehicles[random_destination_vehicle].destinations)

    solution.calculate_length_of_routes(instance)
    solution.calculate_vehicles_loads()
    solution.calculate_routes_time_windows(instance)
    solution.objective_function(instance)

    return solution

def MMOEASA_mutation3(instance: ProblemInstance, solution: FIGASolution) -> FIGASolution:
    # move a random customer in a random vehicle from its original vehicle to another random position in another random vehicle
    random_origin_vehicle = select_random_vehicle(solution)
    origin_position = rand(1, solution.vehicles[random_origin_vehicle].get_num_of_customers_visited())

    random_destination_vehicle = select_random_vehicle(solution, exclude_values=set((random_origin_vehicle,)))
    destination_position = rand(1, solution.vehicles[random_destination_vehicle].get_num_of_customers_visited())

    solution.vehicles[random_destination_vehicle].destinations.insert(destination_position, solution.vehicles[random_origin_vehicle].destinations.pop(origin_position))

    if not solution.vehicles[random_origin_vehicle].get_num_of_customers_visited():
        del solution.vehicles[random_origin_vehicle]

    solution.calculate_length_of_routes(instance)
    solution.calculate_vehicles_loads()
    solution.calculate_routes_time_windows(instance)
    solution.objective_function(instance)

    return solution
