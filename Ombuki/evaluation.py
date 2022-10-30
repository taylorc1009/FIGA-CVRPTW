from Ombuki.ombukiSolution import OmbukiSolution
from problemInstance import ProblemInstance
from typing import Tuple, List

def ref_point(instance: ProblemInstance) -> Tuple[float, int]:
    return 10000.0, instance.amount_of_vehicles + 1

def calculate_Hypervolumes_area(nondominated_set: List[OmbukiSolution], ref_Hypervolumes: Tuple[float, int]) -> float:
    prev_TD, _ = ref_TD, ref_NV = ref_Hypervolumes[0], ref_Hypervolumes[1]
    area = 0.0

    for solution in sorted([s for s in nondominated_set], key=lambda x: x.total_distance, reverse=True):
        area += (prev_TD - solution.total_distance) * (ref_NV - solution.num_vehicles)
        prev_TD, _ = solution.total_distance, solution.num_vehicles

    return (area / (ref_TD * ref_NV)) * 100
