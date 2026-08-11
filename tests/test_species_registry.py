import unittest

from rxn_checker.species import PROPERTY_REGISTRY, SpeciesProperties


class SpeciesRegistryTests(unittest.TestCase):
    def test_species_define_their_atoms(self) -> None:
        self.assertEqual(PROPERTY_REGISTRY.get_record("CH4").atoms, {"C": 1, "H": 4})

    def test_real_species_molecular_weights_share_one_atomic_weight_basis(
        self,
    ) -> None:
        atomic_weights_g_per_mol = {
            "Al": 26.9815386,
            "Ar": 39.948,
            "C": 12.0107,
            "Ca": 40.078,
            "Cu": 63.546,
            "Fe": 55.845,
            "H": 1.00794,
            "He": 4.002602,
            "N": 14.0067,
            "Ni": 58.6934,
            "O": 15.9994,
        }

        for species_id, record in PROPERTY_REGISTRY.records.items():
            if "Ex" in record.atoms:
                continue
            expected = (
                sum(
                    atomic_weights_g_per_mol[element] * count
                    for element, count in record.atoms.items()
                )
                * 1.0e-3
            )
            with self.subTest(species_id=species_id):
                self.assertAlmostEqual(record.mw, expected, places=15)

    def test_noninteger_atom_counts_are_allowed(self) -> None:
        species = SpeciesProperties(
            "Non-stoichiometric oxide", "solid", {"Fe": 0.95, "O": 1}
        )
        self.assertEqual(species.atoms["Fe"], 0.95)

    def test_atoms_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "must define its atoms"):
            SpeciesProperties("Empty", "gas", {})

    def test_phase_must_be_gas_or_solid(self) -> None:
        with self.assertRaisesRegex(ValueError, "phase"):
            SpeciesProperties("Bad phase", "liquid", {"X": 1})

    def test_atom_counts_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite and positive"):
            SpeciesProperties("Bad", "gas", {"C": 0})
