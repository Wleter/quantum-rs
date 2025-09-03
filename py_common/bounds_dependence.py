from dataclasses import dataclass
import json
from typing import Iterable
import numpy as np
import numpy.typing as npt

class BoundsDependence:
    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter="\t") # type: ignore

    @staticmethod
    def parse_json(filename: str, parameter_range: tuple[float, float] | None = None) -> 'BoundsDependence':
        with open(filename, "r") as file:
            data = json.load(file)

        parameters = data['parameters']
        bound_states = [
            BoundStates(
                item['energies'],
                item['nodes'],
                item['occupations'] if "occupations" in item else None,
            )
            for item in data['bound_states']
        ]

        add_size = 0
        if bound_states[0].occupations is not None:
            add_size = len(bound_states[0].occupations[0])

        data = np.zeros((0, 3 + add_size))
        for parameter, bounds in zip(parameters, bound_states):
            if parameter_range is not None and (parameter < parameter_range[0] or parameter > parameter_range[1]):
                continue
            for i, (node, energy) in enumerate(zip(bounds.nodes, bounds.energies)):
                single_bound = [parameter, node, energy]
                if bounds.occupations is not None:
                    single_bound.extend(bounds.occupations[i])

                data = np.append(data, np.array(single_bound).reshape((1, -1)), axis=0)

        instance = BoundsDependence.__new__(BoundsDependence)
        instance.data = data

        return instance
    
    @staticmethod
    def parse_field_json(filename: str, parameter_range: tuple[float, float] | None = None) -> 'BoundsDependence':
        with open(filename, "r") as file:
            data = json.load(file)

        energies = data['energies']
        bound_states = [
            FieldBoundStates(
                item['fields'],
                item['nodes'],
                item['occupations'] if "occupations" in item else None,
            )
            for item in data['bound_states']
        ]

        add_size = 0
        if bound_states[0].occupations is not None:
            add_size = len(bound_states[0].occupations[0])

        data = np.zeros((0, 3 + add_size))
        for energy, bounds in zip(energies, bound_states):
            for i, (node, field) in enumerate(zip(bounds.nodes, bounds.fields)):
                if parameter_range is not None and (field < parameter_range[0] or field > parameter_range[1]):
                    continue
                single_bound = [field, node, energy]
                if bounds.occupations is not None:
                    single_bound.extend(bounds.occupations[i])

                data = np.append(data, np.array(single_bound).reshape((1, -1)), axis=0)

        instance = BoundsDependence.__new__(BoundsDependence)
        instance.data = data

        return instance

    def dependence(self) -> npt.NDArray[np.float64]:
        return self.data[:, [0,2]]
    
    def states(self) -> Iterable[npt.NDArray[np.float64]]:
        states = np.unique(self.data[:, 1])
        for s in states:
            mask = self.data[:, 1] == s
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = np.delete(filtered, 1, axis=1)
            if len(filtered.shape) == 1:
                yield filtered
                continue

            sorted_indices = np.argsort(filtered[:, 0])

            yield filtered[sorted_indices]

    def fields(self) -> Iterable[npt.NDArray[np.float64]]:
        fields = np.unique(self.data[:, 0])
        for f in fields:
            mask = self.data[:, 0] == f
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = np.delete(filtered, 1, axis=1)
            if len(filtered.shape) == 1:
                yield filtered
                continue

            sorted_indices = np.argsort(filtered[:, 0])[::-1]

            yield filtered[sorted_indices]

class BoundsDependence2D:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def parse_json(filename: str) -> 'BoundsDependence2D':
        with open(filename, "r") as file:
            data = json.load(file)

        parameters = data['parameters']
        bound_states = [
            BoundStates(
                item['energies'],
                item['nodes'],
                item['occupations'],
            )
            for item in data['bound_states']
        ]

        add_size = 0
        if bound_states[0].occupations is not None:
            add_size = len(bound_states[0].occupations[0])

        data = np.zeros((0, 4 + add_size))
        for parameter, bounds in zip(parameters, bound_states):
            for i, (node, energy) in enumerate(zip(bounds.nodes, bounds.energies)):
                single_bound = [parameter[0], parameter[1], node, energy]
                if bounds.occupations is not None:
                    single_bound.extend(bounds.occupations[i])

                data = np.append(data, np.array(single_bound).reshape((1, -1)), axis=0)

        return BoundsDependence2D(data)

    def dependence(self) -> npt.NDArray[np.float64]:
        return self.data[:, [0, 1, 3]]
    
    def states(self) -> Iterable[npt.NDArray[np.float64]]:
        states = np.unique(self.data[:, 2])
        for s in states:
            mask = self.data[:, 2] == s
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = filtered[:, [0, 1, 3]]
            if len(filtered.shape) == 1:
                yield filtered
                continue

            yield filtered

    def slice_len(self, axis: int = 1) -> int:
        grid = np.unique(self.data[:, axis])
        return len(grid)

    def slice(self, index: int, axis: int = 1) -> tuple[BoundsDependence, float]:
        assert axis == 0 or axis == 1

        grid = np.unique(self.data[:, axis])
        assert index < len(grid)

        slice = grid[index]
        filtering = self.data[:, axis] == slice

        instance = BoundsDependence.__new__(BoundsDependence)
        instance.data = np.delete(self.data[filtering, :], axis % 2, 1)

        return instance, slice # type: ignore
@dataclass
class BoundStates:
    energies: list[float]
    nodes: list[int]
    occupations: list[list[float]] | None = None

@dataclass
class FieldBoundStates:
    fields: list[float]
    nodes: list[int]
    occupations: list[list[float]] | None = None

if __name__ == "__main__":
    for s in BoundsDependence(f"data/srf_rb_bounds_2.dat").states():
        print(s)