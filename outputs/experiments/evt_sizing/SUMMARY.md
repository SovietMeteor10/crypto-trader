# Experiment B — EVT-based Position Sizing on V3

Date: 2026-03-22 01:47
Runtime: 486s

## Comparison Table

| Config | Train Sharpe | Val Sharpe | Holdout Sharpe | Max DD% | DD days | Active DD | Worst Month | Trades | Flags |
|--------|-------------|-----------|---------------|---------|---------|-----------|-------------|--------|-------|
| V3 fixed frac | 1.7705 | 2.0036 | 0.9065 | 3.22 | 241 | 14 | -1.21% | 133 | long_dd |
| V3+EVT conservative | 1.7731 | 1.9812 | 0.8959 | 0.34 | 241 | 14 | -0.12% | 133 | long_dd |
| V3+EVT moderate | 1.8042 | 1.7902 | 0.8833 | 0.43 | 241 | 14 | -0.15% | 133 | overfit,long_dd |
| V3+EVT aggressive | 1.8010 | 1.8031 | 0.8779 | 0.69 | 241 | 14 | -0.25% | 133 | overfit,long_dd |

## EVT Leverage Statistics (Holdout)

| Config | Mean Lev | Min Lev | Max Lev |
|--------|---------|---------|---------|
| conservative | 0.31 | 0.30 | 0.52 |
| moderate | 0.41 | 0.30 | 0.87 |
| aggressive | 0.66 | 0.50 | 1.40 |

## Best Result

- **Config**: conservative
- **Holdout Sharpe**: 0.8959
- **Max DD%**: 0.34%
- **DD days**: 241
- **Active DD days**: 14
- **Passes all (original)**: False
- **Passes all (active DD)**: True

## Conclusion

Best config: conservative with holdout Sharpe 0.8959. Passes all (active DD): True. Max DD: 0.34%, DD days: 241, Active DD: 14 days.
