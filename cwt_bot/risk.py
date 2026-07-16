from __future__ import annotations

from typing import Dict, Any


def risk_profile_for_execution_tf(tf: str) -> float:
    table = {'5m': 0.15, '15m': 0.22, '30m': 0.30, '1h': 0.40, '4h': 0.60, '1d': 1.00, '1wk': 1.50}
    return table.get(tf, 0.40)


def reversal_continuation_risk_table(analysis_tf: str) -> Dict[str, float | str]:
    table = {
        '1h': {'execution_tf': '15m', 'risk_pct': 0.40, 'reward_pct': 1.20},
        '4h': {'execution_tf': '1h', 'risk_pct': 0.60, 'reward_pct': 1.80},
        '1d': {'execution_tf': '4h', 'risk_pct': 1.00, 'reward_pct': 3.00},
        '1wk': {'execution_tf': '1d', 'risk_pct': 1.50, 'reward_pct': 4.50},
    }
    return table.get(analysis_tf, {'execution_tf': 'custom', 'risk_pct': 0.40, 'reward_pct': 1.20})


def calc_position_size(account_balance: float, risk_pct: float, entry: float | None, stop_loss: float | None, contract_size: float = 1.0) -> Dict[str, Any]:
    if entry is None or stop_loss is None:
        return {'risk_amount': None, 'risk_per_unit': None, 'units': None, 'lot_equivalent': None}
    risk_amount = account_balance * risk_pct / 100.0
    risk_per_unit = abs(float(entry) - float(stop_loss))
    if risk_per_unit <= 0:
        return {'risk_amount': round(risk_amount, 4), 'risk_per_unit': 0.0, 'units': None, 'lot_equivalent': None}
    units = risk_amount / risk_per_unit
    lot_equivalent = units / max(contract_size, 1e-12)
    return {'risk_amount': round(risk_amount, 4), 'risk_per_unit': round(risk_per_unit, 8), 'units': round(units, 4), 'lot_equivalent': round(lot_equivalent, 6)}
