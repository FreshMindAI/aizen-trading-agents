"""Phase 2 GNN coverage report - uses a materialized temp table for speed."""
import sqlite3, time
from src.gnn.constants import UNIVERSE

GNN_COLS = [
    'return_1', 'return_4', 'return_16', 'volatility_16', 'rsi_14', 'macd_pct',
    'hl_range', 'atr_pct_14', 'ma_dist_20', 'ma_dist_50', 'volume_ratio_20',
    'vwap_distance', 'spy_ret_1', 'qqq_ret_past_16',
]

where_all14 = ' AND '.join([f'{c} IS NOT NULL' for c in GNN_COLS])

conn = sqlite3.connect('data/trading.db', timeout=60)

t0 = time.time()
print('materializing view...', flush=True)
conn.execute('DROP TABLE IF EXISTS tmp_v2')
conn.execute('CREATE TABLE tmp_v2 AS SELECT * FROM v_features_underlying_v2')
conn.execute('CREATE INDEX idx_tmp_v2_sym ON tmp_v2(symbol)')
conn.execute('CREATE INDEX idx_tmp_v2_sym_ts ON tmp_v2(symbol, timestamp)')
print(f'  done in {time.time()-t0:.1f}s', flush=True)

print('| symbol | total_rows | all14_populated | pct | min_ts | max_ts | first_all14 | last_all14 |')
print('|--------|-----------:|----------------:|----:|--------|--------|-------------|------------|')

for sym in UNIVERSE:
    tot = conn.execute('SELECT COUNT(*) FROM tmp_v2 WHERE symbol = ?', (sym,)).fetchone()[0]
    all14 = conn.execute(
        f'SELECT COUNT(*) FROM tmp_v2 WHERE symbol = ? AND {where_all14}',
        (sym,)
    ).fetchone()[0]
    min_ts = conn.execute('SELECT MIN(timestamp) FROM tmp_v2 WHERE symbol = ?', (sym,)).fetchone()[0]
    max_ts = conn.execute('SELECT MAX(timestamp) FROM tmp_v2 WHERE symbol = ?', (sym,)).fetchone()[0]
    first_all = conn.execute(
        f'SELECT MIN(timestamp) FROM tmp_v2 WHERE symbol = ? AND {where_all14}',
        (sym,)
    ).fetchone()[0]
    last_all = conn.execute(
        f'SELECT MAX(timestamp) FROM tmp_v2 WHERE symbol = ? AND {where_all14}',
        (sym,)
    ).fetchone()[0]
    pct = (all14 / tot * 100) if tot else 0
    print(f'| {sym:6s} | {tot:>10,d} | {all14:>15,d} | {pct:>3.0f}% | {min_ts} | {max_ts} | {first_all} | {last_all} |')

print()
print('NULL% per column per symbol:')
hdr = 'symbol | ' + ' | '.join(f'{c[:10]:>11s}' for c in GNN_COLS)
print(hdr)

flagged_total = []
for sym in UNIVERSE:
    null_pcts = []
    for c in GNN_COLS:
        n_null = conn.execute(
            f'SELECT COUNT(*) FROM tmp_v2 WHERE symbol = ? AND {c} IS NULL',
            (sym,)
        ).fetchone()[0]
        tot = conn.execute(
            'SELECT COUNT(*) FROM tmp_v2 WHERE symbol = ?', (sym,)
        ).fetchone()[0]
        p = (n_null / tot * 100) if tot else 0
        null_pcts.append(p)
    flagged = [c for c, p in zip(GNN_COLS, null_pcts) if p > 20]
    row = f'{sym:6s} | ' + ' | '.join(f'{p:>10.1f}%' for p in null_pcts)
    print(row)
    if flagged:
        flagged_total.append((sym, flagged))

print()
if flagged_total:
    print('Symbols with >20% NULL in any GNN column:')
    for sym, cols in flagged_total:
        print(f'  {sym}: {cols}')
else:
    print('No GNN column exceeds 20% NULL in any symbol.')

conn.execute('DROP TABLE tmp_v2')
