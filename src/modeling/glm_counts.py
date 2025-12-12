"""
Poisson / Negative Binomial regressions for zone-level counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .poisson_zone import bucket_counts_by_group


@dataclass
class GLMResult:
    model_name: str
    aic: float
    dispersion: float
    coef: pd.Series
    predictions: pd.Series
    residuals: pd.Series


def prepare_features(counts: pd.Series) -> pd.DataFrame:
    df = counts.to_frame("arrivals")
    idx = df.index
    df["hour"] = idx.hour
    df["weekday"] = idx.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["month_day"] = idx.day
    return df


def fit_glm(
    counts: pd.Series,
    family: str = "poisson",
    add_const: bool = True,
) -> GLMResult:
    df = prepare_features(counts)
    formula = "arrivals ~ C(hour) + C(weekday)"
    if family == "poisson":
        model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
    elif family == "nb":
        model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
    else:
        raise ValueError(f"Unsupported family: {family}")
    result = model.fit()
    preds = result.predict(df)
    residuals = df["arrivals"] - preds
    dispersion = result.pearson_chi2 / result.df_resid if result.df_resid > 0 else np.nan
    return GLMResult(
        model_name=family,
        aic=result.aic,
        dispersion=dispersion,
        coef=result.params,
        predictions=pd.Series(preds, index=counts.index),
        residuals=pd.Series(residuals, index=counts.index),
    )


def compare_models(
    counts: pd.Series,
    families: tuple[str, ...] = ("poisson", "nb"),
) -> Dict[str, GLMResult]:
    return {fam: fit_glm(counts, family=fam) for fam in families}
