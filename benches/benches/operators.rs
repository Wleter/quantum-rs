use diol::prelude::*;
use quantum::{
    cast_variant,
    clebsch_gordan::half_integer::{HalfI32, HalfU32},
    operator_mel,
    states::{
        spins::{get_spin_basis, Spin, SpinOperators},
        state::{into_variant, StateBasis},
        States, StatesBasis,
    },
};
use scattering_solver::faer;
use scattering_solver::faer::Mat;

fn main() -> eyre::Result<()> {
    let bench = Bench::new(Config::from_args()?);

    bench.register(
        "simple operator",
        simple_operator,
        [32, 128, 512, 1024, 2048],
    );
    bench.register(
        "simple operator manual",
        simple_manual_operator,
        [32, 128, 512, 1024, 2048],
    );

    bench.register("simple operator", complex_operator, [2, 4, 8, 16]);
    bench.register(
        "simple operator manual",
        complex_manual_operator,
        [2, 4, 8, 16],
    );

    bench.run()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SimpleState {
    Spin(Spin),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ComplexState {
    Spin1(Spin),
    Spin2(Spin),
    Spin3(Spin),
    Spin4(Spin),
}

fn simple_operator(bencher: Bencher, size: u32) {
    let state = into_variant(
        get_spin_basis(HalfU32::from_doubled(size)),
        SimpleState::Spin,
    );
    let mut basis = States::default();
    basis.push_state(StateBasis::new(state));
    let basis = basis.get_basis();

    bencher.bench(|| {
        let mut operator = operator_mel!(&basis, |[s: SimpleState::Spin]| {
            SpinOperators::ladder_minus(s) + SpinOperators::ladder_plus(s) + SpinOperators::proj_z(s)
        });

        black_box(&mut operator);
    });
}

fn simple_manual_operator(bencher: Bencher, size: u32) {
    let s = HalfU32::from_doubled(size);
    let ms = get_spin_basis(s)
        .iter()
        .map(|x| x.ms)
        .collect::<Vec<HalfI32>>();

    bencher.bench(|| {
        let mut operator = Mat::from_fn(ms.len(), ms.len(), |i, j| unsafe {
            let ms_bra = *ms.get_unchecked(i);
            let ms_ket = *ms.get_unchecked(j);

            let mut value = 0.0;

            if ms_bra == ms_ket {
                value += ms_bra.value()
            }

            if ms_bra.double_value() == ms_ket.double_value() + 2
                || ms_bra.double_value() + 2 == ms_ket.double_value()
            {
                value += (s.value() * (s.value() + 1.) - ms_bra.value() * ms_ket.value()).sqrt()
            }

            value
        });

        black_box(&mut operator);
    });
}

fn complex_operator(bencher: Bencher, size: u32) {
    let spins = get_spin_basis(HalfU32::from_doubled(size));

    let mut basis = States::default();
    basis
        .push_state(StateBasis::new(into_variant(
            spins.clone(),
            ComplexState::Spin1,
        )))
        .push_state(StateBasis::new(into_variant(
            spins.clone(),
            ComplexState::Spin2,
        )))
        .push_state(StateBasis::new(into_variant(
            spins.clone(),
            ComplexState::Spin3,
        )))
        .push_state(StateBasis::new(into_variant(spins, ComplexState::Spin4)));

    let basis: StatesBasis<ComplexState> = basis
        .iter_elements()
        .filter(|x| {
            let state1 = cast_variant!(x[0], ComplexState::Spin1);
            let state2 = cast_variant!(x[1], ComplexState::Spin2);
            let state3 = cast_variant!(x[2], ComplexState::Spin3);
            let state4 = cast_variant!(x[3], ComplexState::Spin4);

            (state1.ms + state2.ms + state3.ms + state4.ms).double_value() == 0
        })
        .collect();

    bencher.bench(|| {
        let mut operator = operator_mel!(&basis, |[s2: ComplexState::Spin2, s4: ComplexState::Spin4]| {
            SpinOperators::dot(s2, s4)
        });

        black_box(&mut operator);
    });
}

fn complex_manual_operator(bencher: Bencher, size: u32) {
    let s = HalfU32::from_doubled(size);
    let ms = get_spin_basis(s)
        .iter()
        .map(|x| x.ms)
        .collect::<Vec<HalfI32>>();

    let mut states = vec![];
    for &m1 in &ms {
        for &m2 in &ms {
            for &m3 in &ms {
                for &m4 in &ms {
                    if (m1 + m2 + m3 + m4).double_value() == 0 {
                        states.push([m1, m2, m3, m4]);
                    }
                }
            }
        }
    }

    bencher.bench(|| {
        let mut operator = Mat::from_fn(states.len(), states.len(), |i, j| unsafe {
            let ms_bra = *states.get_unchecked(i);
            let ms_ket = *states.get_unchecked(j);

            if ms_bra[0] != ms_ket[0] || ms_bra[2] != ms_ket[2] {
                return 0.0;
            }

            let mut value = 0.0;

            if ms_bra[1] == ms_ket[1] && ms_bra[3] == ms_ket[3] {
                value += ms_bra[1].value() * ms_bra[3].value()
            }

            if ms_bra[1].double_value() == ms_ket[1].double_value() + 2
                && ms_bra[3].double_value() + 2 == ms_ket[3].double_value()
            {
                value += (s.value() * (s.value() + 1.)
                    - ms_bra[1].value() * ms_ket[1].value() * s.value() * (s.value() + 1.)
                    - ms_bra[3].value() * ms_ket[3].value())
                .sqrt()
            }

            if ms_bra[1].double_value() + 2 == ms_ket[1].double_value()
                && ms_bra[3].double_value() == ms_ket[3].double_value() + 2
            {
                value += (s.value() * (s.value() + 1.)
                    - ms_bra[1].value() * ms_ket[1].value() * s.value() * (s.value() + 1.)
                    - ms_bra[3].value() * ms_ket[3].value())
                .sqrt()
            }

            value
        });

        black_box(&mut operator);
    });
}
