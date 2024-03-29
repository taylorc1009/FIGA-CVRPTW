from typing import List, Tuple

from MMOEASA.mmoeasaSolution import MMOEASASolution
from problemInstance import ProblemInstance


def ref_point(instance: ProblemInstance) -> Tuple[float, float, int]:
    return 10000.0, 2000.0, instance.capacity_of_vehicles + 1

def calculate_Hypervolumes_area(nondominated_set: List[MMOEASASolution], ref_Hypervolumes: Tuple[float, float, int]) -> float:
    prev_TD, _, _ = ref_TD, _, ref_CU = ref_Hypervolumes[0], ref_Hypervolumes[1], ref_Hypervolumes[2]
    area = 0.0

    for solution in sorted([s for s in nondominated_set], key=lambda x: x.total_distance, reverse=True):
        area += (prev_TD - solution.total_distance) * (ref_CU - solution.cargo_unbalance)
        prev_TD, _, _ = solution.total_distance, solution.distance_unbalance, solution.cargo_unbalance

    return (area / (ref_TD * ref_CU)) * 100