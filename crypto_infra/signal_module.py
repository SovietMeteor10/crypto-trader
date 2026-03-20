"""SignalModule — base class only, no strategy logic."""

from abc import ABC, abstractmethod
import pandas as pd


class SignalError(Exception):
    """Raised when signal output is invalid."""


class SignalModule(ABC):
    """
    Base class only. Every strategy implements this interface.
    No trading logic lives here.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @property
    @abstractmethod
    def parameter_space(self) -> dict:
        """
        Defines the Optuna search space for this strategy.
        Format: {param_name: ("int"|"float"|"categorical", low, high)
                              or ("categorical", [list_of_values])}
        """

    @abstractmethod
    def generate(
        self,
        data: pd.DataFrame,
        params: dict,
    ) -> pd.Series:
        """
        Returns Series aligned to data.index with values in {-1, 0, +1}.
        No lookahead bias — signal at time T uses data up to and including T only.
        """

    def validate_output(self, signal: pd.Series, data: pd.DataFrame) -> None:
        """
        Called automatically by BacktestEngine after generate().
        Checks: same index as data, values only in {-1, 0, 1}, no NaN.
        """
        if not signal.index.equals(data.index):
            raise SignalError("Signal index does not match data index")
        valid_values = {-1, 0, 1}
        unique_vals = set(signal.unique())
        if not unique_vals.issubset(valid_values):
            raise SignalError(f"Invalid signal values: {unique_vals - valid_values}")
        if signal.isna().any():
            raise SignalError("NaN values in signal")
