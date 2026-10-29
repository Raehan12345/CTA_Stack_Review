"""Out-of-sample testing.

Two tools, used for different questions.

rolling_folds   train/test splits of a price frame, for re-fitting a rule
                inside each training window (the first walk-forward, which
                showed that re-optimising the lookbacks per fold scored 0.16
                against 0.34 for leaving them fixed).

nested          the test every later decision had to pass. Given several
                variants' daily returns, pick the variant on each training
                window ONLY, apply it to the next unseen window, and chain
                those windows into one track record that could actually
                have been followed. Compare that chain with simply running
                the baseline.

Why nested matters here: widening the menu of variants lowered chained
Sharpe twice (by 0.028 and 0.099), and the nested result has cost roughly
the full-sample gain every time something was tested. A full-sample
comparison is a hypothesis; the nested chain is the result.
"""
from dataclasses import dataclass

import pandas as pd

from .metrics import sharpe


@dataclass
class Fold:
    train: pd.DataFrame
    test: pd.DataFrame
    train_range: tuple
    test_range: tuple


def rolling_folds(df: pd.DataFrame, train_size: int, test_size: int, step: int = None,
                  history_buffer: int = 0) -> list:
    """Rolling folds, sizes in bars. `history_buffer` prepends that many bars
    to each test slice so long lookbacks can warm up; score only from
    `test_range[0]`, never from the slice's first row."""
    step = step or test_size
    n, folds, start = len(df), [], 0
    while start + train_size + test_size <= n:
        train = df.iloc[start:start + train_size]
        buffer_start = max(0, start + train_size - history_buffer)
        test = df.iloc[buffer_start:start + train_size + test_size]
        true_test = df.iloc[start + train_size:start + train_size + test_size]
        folds.append(Fold(train, test, (train.index[0], train.index[-1]),
                          (true_test.index[0], true_test.index[-1])))
        start += step
    return folds


def two_year_windows(index: pd.DatetimeIndex, start) -> list:
    """Two-year test windows on calendar-year boundaries (2009, 2011, ...) from
    `start` to the end of the data, the last one possibly shorter. With the
    project's out-of-sample start of 2008-09-04 that gives nine windows,
    2009-2011 through 2025-2026; the first window's training data is
    2008-09-04 to 2009-01-01."""
    start = pd.Timestamp(start, tz=index.tz) if not isinstance(start, pd.Timestamp) else start
    edges = list(pd.date_range(start, index.max(), freq="2YS"))
    if edges[-1] < index.max():
        edges.append(index.max())
    return list(zip(edges[:-1], edges[1:]))


def nested(rets: dict, folds: list, train_start, rf: pd.Series = None,
           base: str = None, score=sharpe):
    """Pick a variant per fold on training data only, chain the picks.

    rets      {variant name: daily return series}, all on one calendar
    folds     [(test_start, test_end), ...], each trained on
              [train_start, test_start]
    base      the variant to compare against (the first if None)

    Returns (per-fold table, chained returns of the picks, chained returns
    of the baseline over the same windows).
    """
    base = base or next(iter(rets))
    # a string start date and a tz-aware window edge cannot be mixed in one slice
    train_start = pd.Timestamp(train_start)
    if train_start.tzinfo is None and folds and folds[0][0].tzinfo is not None:
        train_start = train_start.tz_localize(folds[0][0].tzinfo)
    rows, chain, base_chain = [], [], []
    for a, b in folds:
        train_scores = {k: score(r.loc[train_start:a], rf) for k, r in rets.items()}
        pick = max(train_scores, key=lambda k: (train_scores[k] if train_scores[k] == train_scores[k]
                                                else -1e9))
        test = rets[pick].loc[a:b]
        rows.append(dict(window=f"{a.date()} -> {b.date()}", chose=pick,
                         oos=score(test, rf), oos_base=score(rets[base].loc[a:b], rf)))
        chain.append(test)
        base_chain.append(rets[base].loc[a:b])
    dedup = lambda s: s[~s.index.duplicated(keep="first")]
    return (pd.DataFrame(rows).set_index("window"),
            dedup(pd.concat(chain)), dedup(pd.concat(base_chain)))
