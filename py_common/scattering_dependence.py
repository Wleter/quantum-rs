from dataclasses import dataclass
import json
import numpy as np
import numpy.typing as npt

@dataclass
class ScatteringObserwables:
    entrance: int
    scattering_length: complex
    elastic_cross_section: float
    inelastic_cross_sections: list[float]

@dataclass
class ScatteringDependence:
    parameters: list[float]
    cross_sections: list[ScatteringObserwables]

    @staticmethod
    def parse_json(filename: str) -> 'ScatteringDependence':
        with open(filename, "r") as file:
            data = json.load(file)

        cross_sections = [
            ScatteringObserwables(
                entrance = item['entrance'],
                scattering_length = complex(*item['scattering_length']),
                elastic_cross_section = item['elastic_cross_section'],
                inelastic_cross_sections = item['inelastic_cross_sections']
            )
            for item in data['observables']
        ]

        return ScatteringDependence(
            parameters=data['parameters'],
            cross_sections=cross_sections
        )
    
    def s_lengths(self) -> npt.NDArray[np.complex64]:
        return np.array(list(map(lambda x: x.scattering_length, self.cross_sections)))
    
    def elastic_cross_sections(self) -> npt.NDArray[np.float64]:
        return np.array(list(map(lambda x: x.elastic_cross_section, self.cross_sections)))
    
    def inelastic_cross_sections(self, channel: int) -> npt.NDArray[np.float64]:
        return np.array(list(map(lambda x: x.inelastic_cross_sections[channel], self.cross_sections)))