import unittest

from api.app import health_check


class TestAPI(unittest.TestCase):
    def test_health_check(self) -> None:
        self.assertEqual(health_check(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
