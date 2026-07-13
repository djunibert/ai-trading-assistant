import unittest

from src.risk.risk_manager import calculate_risk


class CalculateRiskTest(unittest.TestCase):
    def test_buy_signal_with_enough_confidence_allows_trade(self):
        result = calculate_risk(
            signal="BUY",
            account_balance=100000,
            risk_percent=1,
            entry_price=100,
            atr_14=2,
            confidence=0.75,
            risk_reward_ratio=2,
        )

        self.assertTrue(result["trade_allowed"])
        self.assertEqual(result["stop_loss"], 98)
        self.assertEqual(result["take_profit"], 104)
        self.assertEqual(result["risk_amount"], 1000)
        self.assertEqual(result["position_size"], 500)

    def test_no_trade_signal_blocks_position(self):
        result = calculate_risk(
            signal="NO_TRADE",
            account_balance=100000,
            risk_percent=1,
            entry_price=100,
            atr_14=2,
            confidence=0.95,
        )

        self.assertFalse(result["trade_allowed"])
        self.assertEqual(result["position_size"], 0)
        self.assertIsNone(result["stop_loss"])
        self.assertIsNone(result["take_profit"])


if __name__ == "__main__":
    unittest.main()
