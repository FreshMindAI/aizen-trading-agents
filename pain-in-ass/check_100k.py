import json
with open('models/backtest_6w_stocks_100k.json') as f:
    r = json.load(f)
print('=== AGGREGATE (with $100k capital) ===')
for k, v in r['aggregate'].items():
    print(f'  {k:30s}: {v}')
print()
print('=== PER-CYCLE ===')
for c in r['cycles']:
    legs = json.loads(c['predicted_legs_json']) if c.get('predicted_legs_json') else []
    leg_str = ''
    if legs:
        L = legs[0]
        ac = L.get('asset_class', 'option')
        if ac == 'equity':
            notional = L.get('quantity', 0) * (L.get('limit_price') or 0)
            leg_str = f"{ac} {L.get('quantity')}x {L.get('contract_symbol')} @ ${L.get('limit_price', 0):.2f} = ${notional:.0f}"
        else:
            leg_str = f"{ac} {L.get('quantity')}x {L.get('contract_symbol')}"
    print(f"  {c['cycle_as_of']} {c['final_action']:8s} {c['predicted_underlying']:5s} "
          f"fwd_h4={c['forward_return_h4']:+.4f} "
          f"eq_payoff_h4={c['equity_payoff_h4']!s:>8s} | {leg_str}")
