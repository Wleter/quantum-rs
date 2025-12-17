from dataclasses import dataclass
import json
import re
import numpy as np
import numpy.typing as npt
from array import array
from .units import GHZ, ANGS


def parse_wavefunction_file(path, basis_size, max_coeff=None):
    header_re = re.compile(r"WAVEFUNCTION FOR STATE\s+(\d+)\s+AT ENERGY\s+([-+]?\d*\.\d+(?:[eE][+-]?\d+)?)")

    data = {}
    current_state = None
    energy = None

    r_arr = array('d')
    coeffs_arr = array('d')
    occupations_arr = array('d')
    buf = array('d')
    row_length = 1 + basis_size

    coeff_count = (max_coeff + 1) if max_coeff is not None else basis_size

    with open(path, 'r') as f:
        for line in f:
            m = header_re.search(line)
            if m:
                if current_state is not None:
                    npts = len(r_arr)
                    occupations = np.frombuffer(occupations_arr, dtype=float).reshape(npts, coeff_count).copy()

                    data[current_state] = {
                        'energy': energy,
                        'r': np.frombuffer(r_arr, dtype=float).copy() * ANGS,
                        'coeffs': np.frombuffer(coeffs_arr, dtype=float).reshape(npts, coeff_count).copy() / np.sqrt(ANGS),
                        'coeffs_sqr': occupations.copy() / ANGS,
                        'occupations': np.array([np.trapezoid(occupations[:, j], r_arr) for j in range(coeff_count)])
                    }

                current_state = int(m.group(1))
                energy = float(m.group(2))
                r_arr = array('d')
                coeffs_arr = array('d')
                occupations_arr = array('d')
                buf = array('d')
                continue

            if not line or line[0] in ('#', '\n', '\r'):
                continue

            nums = np.fromstring(line, dtype=float, sep=' ')
            if nums.size == 0:
                continue
            buf.extend(nums.tolist())

            while len(buf) >= row_length:
                r_arr.append(buf[0])
                start = 1
                end = 1 + coeff_count
                coeffs_arr.extend(buf[start:end])
                occupations_arr.extend(np.power(buf[start:(end-1)], 2))
                occupations_arr.append(np.sum(np.power(buf[end:], 2)))
                del buf[:row_length]

    if current_state is not None:
        npts = len(r_arr)
        occupations = np.frombuffer(occupations_arr, dtype=float).reshape(npts, coeff_count).copy()

        data[current_state] = {
            'energy': energy,
            'r': np.frombuffer(r_arr, dtype=float).copy() * ANGS,
            'coeffs': np.frombuffer(coeffs_arr, dtype=float).reshape(npts, coeff_count).copy() / np.sqrt(ANGS),
            'coeffs_sqr': occupations.copy() / ANGS,
            'occupations': np.array([np.trapezoid(occupations[:, j], r_arr) for j in range(coeff_count)])
        }

    return data

def parse_wavefunction_field_file(path, basis_size, max_coeff=None):
    header_re = re.compile(r"WAVEFUNCTION FOR STATE\s+(\d+)\s+AT")

    data = []
    current_state = None

    r_arr = array('d')
    coeffs_arr = array('d')
    buf = array('d')
    row_length = 1 + basis_size

    coeff_count = (max_coeff + 1) if max_coeff is not None else basis_size

    with open(path, 'r') as f:
        for line in f:
            m = header_re.search(line)
            if m:
                if current_state is not None:
                    npts = len(r_arr)
                    data.append({
                        'r': np.frombuffer(r_arr, dtype=float).copy() * ANGS,
                        'coeffs': np.frombuffer(coeffs_arr, dtype=float).reshape(npts, coeff_count).copy() / np.sqrt(ANGS)
                    })

                current_state = int(m.group(1))
                r_arr = array('d')
                coeffs_arr = array('d')
                buf = array('d')
                continue

            if not line or line[0] in ('#', '\n', '\r'):
                continue

            nums = np.fromstring(line, dtype=float, sep=' ')
            if nums.size == 0:
                continue
            buf.extend(nums.tolist())

            while len(buf) >= row_length:
                r_arr.append(buf[0])
                start = 1
                end = 1 + coeff_count
                coeffs_arr.extend(buf[start:end])
                del buf[:row_length]

    if current_state is not None:
        npts = len(r_arr)
        data.append({
            'r': np.frombuffer(r_arr, dtype=float).copy() * ANGS,
            'coeffs': np.frombuffer(coeffs_arr, dtype=float).reshape(npts, coeff_count).copy() / np.sqrt(ANGS)
        })

    return data

@dataclass
class WaveFunction:
    energy: float
    distances: npt.NDArray
    values: npt.NDArray
    values_sqr: npt.NDArray
    occupations: npt.NDArray

    def to_dict(self):
        return {
            "energy": self.energy,
            "r": self.distances,
            "coeffs": self.values,
            "coeffs_sqr": self.values_sqr,
            "occupations": self.occupations,
        }

def wavefunction_json(path: str, take: int | None = None) -> dict[int, WaveFunction]:
    with open(path, "r") as file:
        data = json.load(file)

    waves = {}
    for (n, e, w) in zip(data["bounds"]["nodes"], data["bounds"]["energies"], data["waves"]):
        values = np.array(w["values"])
        values_sqr = values**2

        occupations = np.array([np.trapezoid(values_sqr[:, j], w["distances"]) for j in range(values.shape[1])])

        if take is not None and take < values.shape[1]:
            values_sqr_1 = np.sum(values[:, take:], axis = 1)
            values_sqr = values[:, :(take+1)]
            values_sqr[:, -1] = values_sqr_1

            values = values[:, :take]
            occupations_1 = np.sum(occupations[take:])
            occupations = occupations[:take]
            occupations[-1] = occupations_1

        wave = WaveFunction(e / GHZ, np.array(w["distances"]), values, values_sqr, occupations)
        waves[n] = wave

    return waves

if __name__ == '__main__':
    parsed = parse_wavefunction_file('data/wave_function_srf_rb_n_175_singlet_simple_1_00186.output', 176, max_coeff = 5)

    for st, info in parsed.items():
        print(f"State {st}: energy={info['energy']}, points={len(info['r'])}, coeffs_shape={info['coeffs'].shape}")
        print(info['r'][0:10])
        print(info['coeffs'][0:10, :])
