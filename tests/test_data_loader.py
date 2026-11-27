import unittest

from src.data_loader import load_community


class CommunityLoaderTests(unittest.TestCase):
    def test_base_community_has_consistent_hourly_data(self):
        community = load_community("base")

        self.assertEqual(len(community.members), 16)
        self.assertEqual(len(set(community.member_ids)), 16)
        self.assertEqual(len(community.contract.capacity_limit_kW), 24)
        self.assertAlmostEqual(sum(community.contract.share.values()), 1.0)
        for member in community.members:
            self.assertEqual(len(member.energy_price), 24)
            self.assertEqual(len(member.reference_load), 24)

    def test_loading_another_day_updates_profiles_consistently(self):
        monday = load_community("base", day="d1")
        thursday = load_community("base", day="d4")

        self.assertEqual(monday.member_ids, thursday.member_ids)
        self.assertEqual(thursday.day, "d4")
        self.assertNotEqual(tuple(monday.energy_price), tuple(thursday.energy_price))
        self.assertNotEqual(
            tuple(monday.contract.capacity_limit_kW),
            tuple(thursday.contract.capacity_limit_kW),
        )

    def test_unknown_community_reports_a_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Unknown community"):
            load_community("not_a_community")


if __name__ == "__main__":
    unittest.main()
