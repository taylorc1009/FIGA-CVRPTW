import json
import os.path
import re
import sys
from operator import attrgetter
from pathlib import Path
from typing import List

import pandas as pd

from node import Node
from problemInstance import ProblemInstance
from solution import Solution


def open_problem_instance(algorithm: str, filename: str, acceptance_criterion: str) -> ProblemInstance:
    try:
        with open(filename, 'r') as file:
            problem_instance = None
            problem_name = file.readline().strip() # because the problem name is the first line in the text files, this line quickly adds it to a variable (so we can add it to a "ProblemInstance" object later"
            next(file) # skips the first line (containing the problem name), preventing it from being iterated over

            for line in file:
                if line is not None and not re.search('[a-zA-Z]', line): # checks if the current line contains any characters; we don't want to work with any of the alphabetical characters in the text files, only the numbers
                    cur_line = line.split()
                    if cur_line: # prevents any work being done with empty lines (lines that contained only a new line; '\n')
                        if len(cur_line) == 2: # if the current line only contains two numbers then it's the line that holds only the amount of vehicles and vehicles' capacity, so use them to make a "ProblemInstance"
                            problem_instance = ProblemInstance(problem_name, *cur_line, nodes=dict(), acceptance_criterion=acceptance_criterion)
                        else: # if the current line doesn't contain only two values, it will, instead, always contain seven and lines with seven values represent destinations
                            node = Node(*cur_line)
                            problem_instance.nodes[int(node.number)] = node
            if len(problem_instance.nodes) - 1 == 100 and (algorithm == "MMOEASA" or algorithm == "FIGA"):
                with open("solomon_100/hypervolumes.json") as json_file:
                    problem_instance.update_Hypervolumes(*json.load(json_file)[acceptance_criterion][problem_instance.name])

        problem_instance.calculate_distances()
        return problem_instance
    except FileNotFoundError as e:
        exc = FileNotFoundError(f"Couldn't open file \"{filename}\"\nCause: {e}")
        raise exc from None

def MMOEASA_write_solution_for_validation(solution: Solution, max_capacity: int) -> None:
    if getattr(sys, "frozen", False):
        exc = RuntimeError("Attempted to write a solution for validation, but the app appears to be compiled. This is not recommended as compiled apps are considered to be \"production\" apps and this functionality is intended for debugging.")
        raise exc
    relative_path = str(Path(__file__).parent.resolve()) + "\\MMOEASA\\validator\\solution.csv"

    with open(relative_path, "w+") as csv:
        csv.write(f"{max_capacity}\n")
        csv.write(f"{1 if solution.feasible else 0},{solution.total_distance},{solution.distance_unbalance},{solution.cargo_unbalance},{len(solution.vehicles)}\n")
        for vehicle in solution.vehicles:
            csv.write(f"{vehicle.current_capacity},{vehicle.route_distance},{len(vehicle.destinations)}\n")
            for destination in vehicle.destinations:
                csv.write(f"{destination.arrival_time},{destination.departure_time},{destination.wait_time}\n")
                node = destination.node
                csv.write(f"{node.number},{node.x},{node.y},{node.demand},{node.ready_time},{node.due_date},{node.service_duration}\n")

def write_solution_for_graph(solution: Solution) -> None:
    if getattr(sys, "frozen", False):
        exc = RuntimeError("Attempted to write a solution for plotting on a graph, but the app appears to be compiled. This is not recommended as compiled apps are considered to be \"production\" apps and this functionality is intended for non-production purposes.")
        raise exc

    relative_path = str(Path(__file__).parent.resolve()) + f"\\for_graphs\\graph_solution.csv"

    with open(relative_path, "w+") as csv:
        max_len = max([len(v.destinations) for v in solution.vehicles])
        for d in range(max_len):
            for vehicle in solution.vehicles:
                if d < len(vehicle.destinations):
                    csv.write(f"{vehicle.destinations[d].node.x},{vehicle.destinations[d].node.y},")
                else:
                    csv.write(",,")
            csv.write('\n')

def store_results(problem_instance: str, algorithm: str, all_hypervolumes: List[float], all_nondominated_sets: List[List[Solution]], final_hypervolume: float, final_nondominated_set: List[Solution]) -> None:
    relative_path = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else str(Path(__file__).parent.resolve())
    problem_name = re.split("[\/\.]+", problem_instance)[1]

    summary_csv_path = relative_path + "\\summary.csv"
    run_id = len(pd.read_csv(summary_csv_path, header=None)) if os.path.isfile(summary_csv_path) else 0

    with open(summary_csv_path, "a+") as csv:
        v_low, v_high = min(final_nondominated_set, key=attrgetter("num_vehicles")).num_vehicles, max(final_nondominated_set, key=attrgetter("num_vehicles")).num_vehicles
        d_low, d_high = min(final_nondominated_set, key=attrgetter("total_distance")).total_distance, max(final_nondominated_set, key=attrgetter("total_distance")).total_distance
        csv.write(f"{run_id},{problem_name},{algorithm},{final_hypervolume},{len(final_nondominated_set)},{d_low},{d_high},{v_low},{v_high}\n")

    with open(relative_path + "\\fronts.csv", "a+") as csv:
        for solution in final_nondominated_set:
            csv.write(f"{run_id},{problem_name},{algorithm},{solution.total_distance},{solution.num_vehicles}\n")

    with open(relative_path + f"\\{algorithm}-{problem_name}.csv", "a+") as csv:
        for i, nondominated_set in enumerate(all_nondominated_sets):
            line = f"{run_id}.{i},{all_hypervolumes[i]},"
            for j, solution in enumerate(nondominated_set, 1):
                solution_str = None
                if hasattr(solution, "num_vehicles"):
                    solution_str = f"{solution.total_distance} {solution.num_vehicles}"
                elif hasattr(solution, "distance_unbalance") and hasattr(solution, "cargo_unbalance"):
                    solution_str = f"{solution.total_distance} {solution.distance_unbalance} {solution.cargo_unbalance}"
                line += solution_str + str([str([d.node.number for d in v.get_customers_visited()]).replace(", ", " ") for v in solution.vehicles]).replace(", ", "").replace("'", "")
                if j < len(nondominated_set):
                    line += ','
            line += '\n'
            csv.write(line)
