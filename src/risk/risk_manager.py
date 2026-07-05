def calculate_risk(
    signal: str,
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    atr_14: float,
    confidence: float,
    risk_reward_ratio: float = 2.0,
    min_confidence: float = 0.60,
):
    risk_amount = account_balance * (risk_percent / 100)

    trade_allowed = signal in ["BUY", "SELL"] and confidence >= min_confidence

    if not trade_allowed:
        return {
            "trade_allowed": False,
            "reason": "Signal faible ou NO_TRADE",
            "risk_amount": risk_amount,
            "position_size": 0,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward_ratio": risk_reward_ratio,
        }

    stop_distance = atr_14

    if signal == "BUY":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (stop_distance * risk_reward_ratio)
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - (stop_distance * risk_reward_ratio)

    position_size = risk_amount / stop_distance if stop_distance > 0 else 0

    return {
        "trade_allowed": True,
        "signal": signal,
        "entry_price": entry_price,
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "risk_amount": round(risk_amount, 2),
        "position_size": round(position_size, 4),
        "risk_reward_ratio": risk_reward_ratio,
        "confidence": confidence,
    }
