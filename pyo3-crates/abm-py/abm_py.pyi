from typing import Iterable

BOHR_MAG: float
"""
Bohr magneton value in a.u. / G
"""

NUCLEAR_MAG: float
"""
Nuclear magneton value in a.u. / G
"""

class HifiProblemBuilder:
    """
    Builder for Hyperfine Zeeman diagonalization problem
    with provided double electron spin s and double nuclear spin i.
    """
    def __init__(self, double_s: int, double_i: int) -> None: ...

    def with_projection(self, double_projection: int) -> None:
        """
        Solve only for provided total double projection m_f. 
        """

    def with_custom_bohr_magneton(self, gamma_e: float) -> None:
        """
        Set custom bohr magneton value, default is a physical value.
        """

    def with_nuclear_magneton(self, gamma_i: float) -> None:
        """
        Set nuclear magneton value, default is 0.
        """

    def with_hyperfine_coupling(self, a_hifi: float) -> None:
        """
        Set hyperine coupling constant, default is 0.
        """

    def build(self) -> HifiProblem:
        """
        Builds the hyperfine Zeeman problem with specified problem values.
        """

        
class DoubleHifiProblemBuilder:
    """
    Builder for Hyperfine Zeeman diagonalization problem
    for first and second atoms described by HifiProblemBuilder.
    For homonuclear case use new_homo static method
    """
    def __init__(self, first: HifiProblemBuilder, second: HifiProblemBuilder) -> None: ...

    @staticmethod
    def new_homo(single: HifiProblemBuilder, symmetry: str = "none") -> DoubleHifiProblemBuilder:
        """
        Creates DoubleHifiProblemBuilder for homonuclear case with atom 
        specified by HifiProblemBuilder and symmetry parameter.

        Allowed values of symmetry are "none", "fermionic", "bosonic".
        """

    def with_projection(self, double_projection: int) -> None:
        """
        Solve only for provided total double projection m_F. 
        """

    def build(self) -> HifiProblem:
        """
        Builds the hyperfine Zeeman problem with specified problem values.
        """

class ABMProblemBuilder:
    """
    Builder for Hyperfine Zeeman diagonalization problem
    for first and second atoms described by HifiProblemBuilder.
    For homonuclear case use new_homo static method
    """
    def __init__(self, first: HifiProblemBuilder, second: HifiProblemBuilder) -> None: ...

    @staticmethod
    def new_homo(single: HifiProblemBuilder, symmetry: str = "none") -> ABMProblemBuilder:
        """
        Creates ABMProblemBuilder for homonuclear case with atom 
        specified by HifiProblemBuilder and symmetry parameter.

        Allowed values of symmetry are "none", "fermionic", "bosonic".
        """

    def with_projection(self, double_projection: int) -> None:
        """
        Solve only for provided total double projection m_F. 
        """

    def with_vibrational(self, singlet_energies: Iterable[float], triplet_energies: Iterable[float], fc_factors: Iterable[float]) -> None:
        """
        Solve for provided singlet states energies, triplet states energies and Franck-Condon factors between singlet, triplet states.

        Franck-Condon factors should be provided by the flattened version of the <S = 0, v_S | S = 1, v_S> matrix.
        """

    def build(self) -> HifiProblem:
        """
        Builds the Aymptotic Bound-state model problem with specified problem values.
        """

class HifiProblem:
    """
    Hyperfine Zeeman problem solver for different magnetic fields.
    """

    def states_at(self, magnetic_field: float) -> list[float]:
        """
        Return eigenstates of the system at specified magnetic field.
        """

    def states_range(self, magnetic_fields: Iterable[float]) -> list[list[float]]:
        """
        Return eigenstates of the system 
        for each magnetic field value in a magnetic_fields list
        """

    
class ABMProblem:
    """
    ABM problem solver for different magnetic fields.
    """

    def states_at(self, magnetic_field: float) -> list[float]:
        """
        Return eigenstates of the system at specified magnetic field.
        """

    def states_range(self, magnetic_fields: Iterable[float]) -> list[list[float]]:
        """
        Return eigenstates of the system 
        for each magnetic field value in a magnetic_fields list
        """

    