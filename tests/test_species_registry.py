from __future__ import annotations

import unittest

from rxn_checker.species import PROPERTY_REGISTRY, SpeciesProperties


class SpeciesRegistryTests(unittest.TestCase):
    def test_species_define_their_atoms(self) -> None:
        self.assertEqual(PROPERTY_REGISTRY.get_record("CH4").atoms, {"C": 1, "H": 4})

    def test_noninteger_atom_counts_are_allowed(self) -> None:
        species = SpeciesProperties("Non-stoichiometric oxide", {"Fe": 0.95, "O": 1})
        self.assertEqual(species.atoms["Fe"], 0.95)

    def test_atoms_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "must define its atoms"):
            SpeciesProperties("Empty", {})

    def test_atom_counts_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite and positive"):
            SpeciesProperties("Bad", {"C": 0})


if __name__ == "__main__":
    unittest.main()
