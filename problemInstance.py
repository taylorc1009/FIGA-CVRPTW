from typing import Dict, List, Union

from node import Node


class ProblemInstance:
    distances: List[float]=None
    Hypervolume_total_distance: float=0.0
    Hypervolume_distance_unbalance: float=0.0 # currently, the distance unbalance objective is unused in the objective function, but this may change
    Hypervolume_cargo_unbalance: float=0.0
    Hypervolume_num_vehicles: float=0.0

    def __init__(self, name: str, amount_of_vehicles: int, capacity_of_vehicles: int, nodes: Dict[int, Node]=None, acceptance_criterion: str="") -> None:
        self.name: str=name
        self.amount_of_vehicles: int=int(amount_of_vehicles)
        self.capacity_of_vehicles: int=int(capacity_of_vehicles)
        self.nodes: Dict[int, Node]=nodes
        self.acceptance_criterion: str=str(acceptance_criterion)

    def __str__(self) -> str:
        return f"ProblemInstance(name={self.name}, amount_of_vehicles={self.amount_of_vehicles}, capacity_of_vehicles={self.capacity_of_vehicles}, {len(self.nodes)=}"#, {[(key, str(value)) for key, value in self.nodes.items()]})"
    
    def calculate_distances(self) -> None:
        n = len(self.nodes)
        self.distances: List[float]=[-1.0 for _ in range(0, n ** 2)]

        for i in range(0, n):
            for j in range(0, n):
                if i != j:
                    self.distances[n * i + j] = self.nodes[i].get_distance(self.nodes[j].x, self.nodes[j].y)

    def get_distance(self, from_node: int, to_node: int) -> float:
        return self.distances[len(self.nodes) * from_node + to_node]

    def update_Hypervolumes(self, *args: List[Union[float, int]]) -> None:
        self.Hypervolume_total_distance = float(args[0])
        if self.acceptance_criterion == "MMOEASA":
            self.Hypervolume_distance_unbalance = float(args[1])
            self.Hypervolume_cargo_unbalance = float(args[2])
            print(f"Hypervolumes modified: TD={self.Hypervolume_total_distance}, DU={self.Hypervolume_distance_unbalance}, CU={self.Hypervolume_cargo_unbalance}")
        elif self.acceptance_criterion == "Ombuki":
            self.Hypervolume_num_vehicles = int(args[1])
            print(f"Hypervolumes modified: TD={self.Hypervolume_total_distance}, NV={self.Hypervolume_num_vehicles}")
