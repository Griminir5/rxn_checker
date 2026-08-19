"""Exact box and phase-chamfered concentration domains."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import sympy as sp

from .model import CaseSymbols, parse_rational


GAS_CONSTANT = sp.Rational("8.31446261815324")


def _exact(value: object) -> sp.Expr:
    expression = sp.sympify(value)
    floats = {item: parse_rational(item) for item in expression.atoms(sp.Float)}
    return expression.xreplace(floats)


class DomainKind(StrEnum):
    PHYSICAL = "physical"
    AUGMENTED = "augmented"


class ConcentrationModel(StrEnum):
    INDEPENDENT = "independent"
    CHAMFERED = "chamfered"


@dataclass(frozen=True)
class Interval:
    lower: sp.Expr
    upper: sp.Expr
    lower_closed: bool = True
    upper_closed: bool = True

    def __post_init__(self) -> None:
        lower, upper = _exact(self.lower), _exact(self.upper)
        if lower.is_real is not True or upper.is_real is not True:
            raise ValueError("Interval bounds must be real.")
        if lower.is_finite is not True or upper.is_finite is not True:
            raise ValueError("Interval bounds must be finite.")
        if not isinstance(self.lower_closed, bool) or not isinstance(
            self.upper_closed, bool
        ):
            raise ValueError("Interval closure flags must be boolean.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, value: object) -> bool:
        value = _exact(value)
        above = value > self.lower or (self.lower_closed and value == self.lower)
        below = value < self.upper or (self.upper_closed and value == self.upper)
        return bool(above and below)


@dataclass(frozen=True)
class TotalConstraint:
    name: str
    symbols: tuple[sp.Symbol, ...]
    minimum: sp.Expr

    def __post_init__(self) -> None:
        symbols = tuple(self.symbols)
        if not self.name or not symbols or len(symbols) != len(set(symbols)):
            raise ValueError("A total constraint needs a name and unique symbols.")
        minimum = _exact(self.minimum)
        if minimum.is_finite is not True or minimum.is_real is not True or minimum <= 0:
            raise ValueError("A total constraint minimum must be finite and positive.")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "minimum", minimum)


@dataclass(frozen=True)
class AffineForm:
    constant: sp.Expr
    coefficients: Mapping[sp.Symbol, sp.Expr]

    def value(self, point: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
        return self.constant + sum(
            coefficient * point[symbol]
            for symbol, coefficient in self.coefficients.items()
        )


@cache
def affine_form(expression: sp.Expr) -> AffineForm | None:
    """Extract an affine form recursively without constructing a polynomial."""

    expression = _exact(expression)
    if not expression.free_symbols:
        return AffineForm(expression, {})
    if isinstance(expression, sp.Symbol):
        return AffineForm(sp.S.Zero, {expression: sp.S.One})
    if isinstance(expression, sp.Add):
        constant = sp.S.Zero
        coefficients: dict[sp.Symbol, sp.Expr] = {}
        for argument in expression.args:
            form = affine_form(argument)
            if form is None:
                return None
            constant += form.constant
            for symbol, coefficient in form.coefficients.items():
                coefficients[symbol] = coefficients.get(symbol, sp.S.Zero) + coefficient
        return AffineForm(
            constant,
            {symbol: value for symbol, value in coefficients.items() if value != 0},
        )
    if isinstance(expression, sp.Mul):
        scale = sp.S.One
        variable_form: AffineForm | None = None
        for argument in expression.args:
            form = affine_form(argument)
            if form is None:
                return None
            if form.coefficients:
                if variable_form is not None:
                    return None
                variable_form = form
            else:
                scale *= form.constant
        if variable_form is None:
            return AffineForm(scale, {})
        return AffineForm(
            scale * variable_form.constant,
            {
                symbol: scale * coefficient
                for symbol, coefficient in variable_form.coefficients.items()
            },
        )
    return None


@dataclass(frozen=True)
class AffineBounds:
    lower: sp.Expr
    upper: sp.Expr
    lower_witness: Mapping[sp.Symbol, sp.Expr] | None
    upper_witness: Mapping[sp.Symbol, sp.Expr] | None


@dataclass(frozen=True)
class ConcentrationDomain:
    kind: DomainKind
    intervals: Mapping[sp.Symbol, Interval]
    parameter_intervals: Mapping[sp.Symbol, Interval]
    total_constraints: tuple[TotalConstraint, ...] = ()

    @property
    def all_intervals(self) -> dict[sp.Symbol, Interval]:
        return {**self.intervals, **self.parameter_intervals}

    def interval(self, symbol: sp.Symbol) -> Interval:
        try:
            return self.intervals.get(symbol) or self.parameter_intervals[symbol]
        except KeyError as error:
            raise KeyError(f"Domain has no symbol '{symbol}'.") from error

    def restrict(
        self,
        symbol: sp.Symbol,
        *,
        lower: object | None = None,
        upper: object | None = None,
        strict_lower: bool = False,
        strict_upper: bool = False,
    ) -> "ConcentrationDomain":
        current = self.interval(symbol)
        new_lower = current.lower if lower is None else _exact(lower)
        new_upper = current.upper if upper is None else _exact(upper)

        if new_lower < current.lower:
            new_lower, lower_closed = current.lower, current.lower_closed
        elif new_lower == current.lower:
            lower_closed = current.lower_closed and not strict_lower
        else:
            lower_closed = not strict_lower
        if new_upper > current.upper:
            new_upper, upper_closed = current.upper, current.upper_closed
        elif new_upper == current.upper:
            upper_closed = current.upper_closed and not strict_upper
        else:
            upper_closed = not strict_upper

        replacement = Interval(new_lower, new_upper, lower_closed, upper_closed)
        concentrations = dict(self.intervals)
        parameters = dict(self.parameter_intervals)
        (concentrations if symbol in concentrations else parameters)[symbol] = replacement
        return ConcentrationDomain(
            self.kind,
            concentrations,
            parameters,
            self.total_constraints,
        )

    def is_feasible(self) -> bool:
        intervals = self.all_intervals
        for interval in intervals.values():
            if interval.lower > interval.upper:
                return False
            if interval.lower == interval.upper and not (
                interval.lower_closed and interval.upper_closed
            ):
                return False
        for constraint in self.total_constraints:
            maximum = sum(intervals[symbol].upper for symbol in constraint.symbols)
            if maximum < constraint.minimum:
                return False
            if maximum == constraint.minimum and any(
                not intervals[symbol].upper_closed for symbol in constraint.symbols
            ):
                return False
        return True

    def exact_witness(
        self,
        preferences: Mapping[sp.Symbol, object] | None = None,
    ) -> dict[sp.Symbol, sp.Expr] | None:
        if not self.is_feasible():
            return None
        preferences = preferences or {}
        intervals = self.all_intervals
        point: dict[sp.Symbol, sp.Expr] = {}
        for symbol, interval in intervals.items():
            preferred = preferences.get(symbol)
            if preferred is not None and interval.contains(preferred):
                point[symbol] = _exact(preferred)
            elif interval.lower_closed:
                point[symbol] = interval.lower
            else:
                point[symbol] = (interval.lower + interval.upper) / 2

        for constraint in self.total_constraints:
            current = sum(point[symbol] for symbol in constraint.symbols)
            remaining = constraint.minimum - current
            if remaining <= 0:
                continue
            capacities = {
                symbol: intervals[symbol].upper - point[symbol]
                for symbol in constraint.symbols
            }
            capacity = sum(capacities.values())
            if remaining > capacity:
                return None
            if remaining == capacity:
                if any(
                    capacities[symbol] and not intervals[symbol].upper_closed
                    for symbol in constraint.symbols
                ):
                    return None
                for symbol in constraint.symbols:
                    point[symbol] = intervals[symbol].upper
            else:
                for symbol, available in capacities.items():
                    point[symbol] += remaining * available / capacity
        return point

    def _extreme(
        self,
        form: AffineForm,
        *,
        maximize: bool,
    ) -> tuple[sp.Expr, Mapping[sp.Symbol, sp.Expr] | None]:
        intervals = self.all_intervals
        unknown = set(form.coefficients) - set(intervals)
        if unknown:
            raise ValueError(
                "Affine expression uses symbols outside the domain: "
                + ", ".join(sorted(map(str, unknown)))
                + "."
            )

        point: dict[sp.Symbol, sp.Expr] = {}
        for symbol, interval in intervals.items():
            coefficient = form.coefficients.get(symbol, sp.S.Zero)
            if coefficient.is_positive is None and coefficient.is_negative is None:
                raise ValueError(f"Cannot determine affine coefficient sign: {coefficient}.")
            prefer_upper = coefficient > 0 if maximize else coefficient < 0
            point[symbol] = interval.upper if prefer_upper else interval.lower

        for constraint in self.total_constraints:
            remaining = constraint.minimum - sum(
                point[symbol] for symbol in constraint.symbols
            )
            if remaining <= 0:
                continue
            candidates = sorted(
                constraint.symbols,
                key=lambda symbol: (
                    form.coefficients.get(symbol, sp.S.Zero),
                    symbol.name,
                ),
                reverse=maximize,
            )
            for symbol in candidates:
                amount = min(remaining, intervals[symbol].upper - point[symbol])
                point[symbol] += amount
                remaining -= amount
                if remaining == 0:
                    break

        value = form.value(point)
        witness = self.exact_witness(point)
        if witness is None or form.value(witness) != value:
            witness = None
        return value, witness

    def affine_bounds(self, expression: sp.Expr) -> AffineBounds | None:
        if not self.is_feasible():
            raise ValueError("Cannot bound an expression on an empty domain.")
        form = affine_form(sp.sympify(expression))
        if form is None:
            return None
        lower, lower_witness = self._extreme(form, maximize=False)
        upper, upper_witness = self._extreme(form, maximize=True)
        return AffineBounds(lower, upper, lower_witness, upper_witness)


@dataclass(frozen=True)
class DomainSpec:
    """One exact specification that generates both domain variants."""

    symbols: CaseSymbols
    concentration_model: ConcentrationModel
    upper: Mapping[sp.Symbol, sp.Expr]
    excursion_lower: Mapping[sp.Symbol, sp.Expr]
    parameter_intervals: Mapping[sp.Symbol, Interval]
    total_constraints: tuple[TotalConstraint, ...] = ()

    def __post_init__(self) -> None:
        model = ConcentrationModel(self.concentration_model)
        upper = {symbol: _exact(value) for symbol, value in self.upper.items()}
        excursion = {
            symbol: _exact(value) for symbol, value in self.excursion_lower.items()
        }
        concentration_symbols = self.symbols.concentration_symbols
        if set(upper) != concentration_symbols or set(excursion) != concentration_symbols:
            raise ValueError("Domain bounds must cover every concentration symbol.")
        if any(value <= 0 for value in upper.values()):
            raise ValueError("Concentration upper bounds must be positive.")
        if any(value > 0 for value in excursion.values()):
            raise ValueError("Excursion lower bounds must be non-positive.")
        if set(self.parameter_intervals) != self.symbols.parameter_symbols:
            raise ValueError("Domain must bound temperature and pressure.")
        if any(
            interval.lower >= interval.upper
            for interval in self.parameter_intervals.values()
        ):
            raise ValueError("Parameter interval bounds must be correctly ordered.")
        if self.parameter_intervals[self.symbols.temperature].lower <= 0:
            raise ValueError("Temperature lower bound must be positive.")

        constraints = tuple(self.total_constraints)
        used: set[sp.Symbol] = set()
        for constraint in constraints:
            if not set(constraint.symbols) <= concentration_symbols:
                raise ValueError("Total constraints may contain only concentrations.")
            if used.intersection(constraint.symbols):
                raise ValueError("Total concentration groups must not overlap.")
            used.update(constraint.symbols)
        if model is ConcentrationModel.INDEPENDENT and constraints:
            raise ValueError("Independent concentration domains cannot define totals.")

        object.__setattr__(self, "concentration_model", model)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "excursion_lower", excursion)
        object.__setattr__(self, "parameter_intervals", dict(self.parameter_intervals))
        object.__setattr__(self, "total_constraints", constraints)
        if not self.build(DomainKind.PHYSICAL).is_feasible():
            raise ValueError("Physical concentration domain is empty.")
        if not self.build(DomainKind.AUGMENTED).is_feasible():
            raise ValueError("Augmented concentration domain is empty.")

    def build(self, kind: DomainKind) -> ConcentrationDomain:
        kind = DomainKind(kind)
        lower = (
            {symbol: sp.S.Zero for symbol in self.upper}
            if kind is DomainKind.PHYSICAL
            else self.excursion_lower
        )
        intervals = {
            symbol: Interval(lower[symbol], self.upper[symbol])
            for symbol in self.upper
        }
        return ConcentrationDomain(
            kind,
            intervals,
            self.parameter_intervals,
            self.total_constraints,
        )


__all__ = (
    "AffineBounds",
    "AffineForm",
    "ConcentrationDomain",
    "ConcentrationModel",
    "DomainKind",
    "DomainSpec",
    "GAS_CONSTANT",
    "Interval",
    "TotalConstraint",
    "affine_form",
)
