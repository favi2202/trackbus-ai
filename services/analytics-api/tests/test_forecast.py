import unittest

from trackbus_analytics.forecast import ForecastInput, forecast_occupancy


class ForecastTests(unittest.TestCase):
    def test_rush_hour_prediction_is_bounded_by_capacity(self) -> None:
        result = forecast_occupancy(
            ForecastInput(
                recent_occupancy=[42, 47, 51, 56, 61, 66],
                capacity=72,
                hour=19,
                weather="rain",
                event_nearby=True,
            )
        )
        self.assertLessEqual(result.expected_occupancy, 72)
        self.assertGreaterEqual(result.lower_bound, 0)
        self.assertLessEqual(result.upper_bound, 72)
        self.assertEqual(result.confidence, 0.86)

    def test_requires_history(self) -> None:
        with self.assertRaises(ValueError):
            forecast_occupancy(ForecastInput(recent_occupancy=[], capacity=72, hour=12))


if __name__ == "__main__":
    unittest.main()
