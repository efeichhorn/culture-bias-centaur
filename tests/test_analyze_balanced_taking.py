"""Sampling and pairing safeguards for the balanced Taking reanalysis."""
import unittest

import numpy as np
import pandas as pd

from scripts.analyze_balanced_taking import (
    KEYS, bootstrap_outcomes, score_contrasts, select_balanced_participants,
    subset_people,
)


class BalancedTakingTests(unittest.TestCase):
    @staticmethod
    def population():
        rows = []
        for dataset, country_counts in ((3, (546, 529)), (4, (825, 800))):
            for nation, count in enumerate(country_counts):
                for participant in range(count):
                    # Two repeated observations are sufficient to test selection
                    # of complete people independently of their response values.
                    for item in (1, 2):
                        rows.append((dataset, nation, participant, item, participant % 21))
        return pd.DataFrame(rows, columns=KEYS + ["GroupNo", "TokenChoice"])

    def test_selection_is_deterministic_whole_person_and_response_independent(self):
        population = self.population()
        selected = select_balanced_participants(population, 20261029)
        altered = population.sample(frac=1, random_state=31).assign(TokenChoice=999)
        pd.testing.assert_frame_equal(selected, select_balanced_participants(altered, 20261029))
        self.assertFalse(selected.equals(select_balanced_participants(population, 20261030)))
        self.assertEqual(selected.groupby(["Dataset", "Nation"]).size().to_dict(),
                         {(3, 0): 529, (3, 1): 529, (4, 0): 800, (4, 1): 800})
        china = population.loc[population.Nation.eq(1), KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(selected.loc[selected.Nation.eq(1)].reset_index(drop=True), china)
        subset = subset_people(population, selected)
        self.assertTrue(subset.groupby(KEYS).size().eq(2).all())

    def test_selection_rejects_missing_population_or_people(self):
        population = self.population()
        with self.assertRaisesRegex(ValueError, "Unexpected Taking population count"):
            select_balanced_participants(population.loc[~(population.Nation.eq(1) & population.SubNo.eq(0))], 0)
        selected = select_balanced_participants(population, 0)
        with self.assertRaisesRegex(ValueError, "missing from an evaluation"):
            subset_people(population.loc[population.Nation.eq(0)], selected)
        with self.assertRaisesRegex(ValueError, "duplicate participants"):
            subset_people(population, pd.concat([selected, selected.iloc[:1]]))

    def test_model_pairing_rejects_missing_participant_condition(self):
        rows = []
        for nation, country in ((0, "United States"), (1, "China")):
            for condition in ("base", "us", "pooled", "china", "us_prior", "pooled_prior", "china_prior"):
                for person in (1, 2):
                    rows.append(dict(Dataset=3, Nation=nation, SubNo=person, seed=0,
                                     condition=condition, Country=country,
                                     conditional_ll=-2.0, raw_response_ll=-2.1))
        complete = pd.DataFrame(rows)
        score_contrasts(complete)
        with self.assertRaisesRegex(ValueError, "identical participants"):
            score_contrasts(complete.iloc[1:])

    def test_shared_bootstrap_preserves_paired_outcomes_and_country_sign(self):
        rows = []
        for nation, country in ((0, "United States"), (1, "China")):
            for seed in range(3):
                for dataset, count in ((3, 3), (4, 5)):
                    for person in range(count):
                        value = float(person + dataset + seed + nation * 2)
                        for contrast, x in (("a", value), ("b", value + 0.5)):
                            rows.append(dict(Dataset=dataset, Nation=nation, SubNo=person,
                                             seed=seed, Country=country, metric="conditional_ll",
                                             contrast=contrast, value=x))
        values = pd.DataFrame(rows)
        intervals, countries = bootstrap_outcomes(values, 100, 42)
        repeated, repeated_countries = bootstrap_outcomes(values, 100, 42)
        pd.testing.assert_frame_equal(intervals, repeated)
        pd.testing.assert_frame_equal(countries, repeated_countries)
        for (_, _), frame in intervals.groupby(["Country", "method"]):
            indexed = frame.set_index("contrast")
            np.testing.assert_allclose(indexed.loc["b", ["mean", "ci_low", "ci_high"]].to_numpy(float)
                                       - indexed.loc["a", ["mean", "ci_low", "ci_high"]].to_numpy(float), .5)
        np.testing.assert_allclose(countries["mean"], -2.0)


if __name__ == "__main__":
    unittest.main()
