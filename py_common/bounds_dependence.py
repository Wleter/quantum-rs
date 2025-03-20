from typing import Iterable
import numpy as np
import numpy.typing as npt

class BoundsDependence:
    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter="\t", skiprows=1)

    def dependence(self) -> npt.NDArray[np.float64]:
        return self.data[:, [0,2]]
    
    def states(self) -> Iterable[npt.NDArray[np.float64]]:
        states = np.unique(self.data[:, 1])
        for s in states:
            mask = self.data[:, 1] == s
            
            if not np.any(mask):
                continue

            filtered = self.data[mask]
            filtered = filtered[:, [0, 2]]
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
            filtered = filtered[:, [0, 2]]
            if len(filtered.shape) == 1:
                yield filtered
                continue

            sorted_indices = np.argsort(filtered[:, 0])[::-1]

            yield filtered[sorted_indices]

if __name__ == "__main__":
    for s in BoundsDependence(f"data/srf_rb_bounds_2.dat").states():
        print(s)