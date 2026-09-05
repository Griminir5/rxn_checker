"""Exact box and phase-chamfered concentration domains."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import sympy as sp

from .model import CaseSymbols, exact_expr

GAS_CONSTANT = sp.Rational("8.31446261815324")


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

    def __post_init__(self):
        lower, upper = exact_expr(self.lower), exact_expr(self.upper)
        if any(
            value.is_real is not True or value.is_finite is not True for value in (lower, upper)
        ):
            raise ValueError("Interval bounds must be finite and real.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def contains(self, value):
        value = exact_expr(value)
        return bool(
            (value > self.lower or self.lower_closed and value == self.lower)
            and (value < self.upper or self.upper_closed and value == self.upper)
        )


@dataclass(frozen=True)
class TotalConstraint:
    name: str
    symbols: tuple[sp.Symbol, ...]
    minimum: sp.Expr

    def __post_init__(self):
        minimum = exact_expr(self.minimum)
        if not self.name or not self.symbols or len(self.symbols) != len(set(self.symbols)):
            raise ValueError("A total constraint needs a name and unique symbols.")
        if minimum.is_finite is not True or minimum.is_real is not True or minimum <= 0:
            raise ValueError("A total constraint minimum must be finite and positive.")
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "minimum", minimum)


@dataclass(frozen=True)
class AffineForm:
    constant: sp.Expr
    coefficients: Mapping[sp.Symbol, sp.Expr]

    def value(self, point):
        return self.constant + sum(
            value * point[symbol] for symbol, value in self.coefficients.items()
        )


@cache
def affine_form(expression) -> AffineForm | None:
    """Extract an affine form without constructing a polynomial."""
    expression = exact_expr(expression)
    if not expression.free_symbols:
        return AffineForm(expression, {})
    if isinstance(expression, sp.Symbol):
        return AffineForm(0, {expression: sp.S.One})
    if isinstance(expression, sp.Add):
        constant, coefficients = sp.S.Zero, {}
        for argument in expression.args:
            form = affine_form(argument)
            if form is None:
                return None
            constant += form.constant
            for symbol, value in form.coefficients.items():
                coefficients[symbol] = coefficients.get(symbol, 0) + value
        return AffineForm(constant, {key: value for key, value in coefficients.items() if value})
    if isinstance(expression, sp.Mul):
        scale, variable = sp.S.One, None
        for argument in expression.args:
            form = affine_form(argument)
            if form is None:
                return None
            if form.coefficients:
                if variable is not None:
                    return None
                variable = form
            else:
                scale *= form.constant
        if variable is None:
            return AffineForm(scale, {})
        return AffineForm(
            scale * variable.constant,
            {key: scale * value for key, value in variable.coefficients.items()},
        )
    return None


@dataclass(frozen=True)
class AffineBounds:
    lower: sp.Expr
    upper: sp.Expr
    lower_witness: Mapping[sp.Symbol, sp.Expr] | None
    upper_witness: Mapping[sp.Symbol, sp.Expr] | None


@dataclass(frozen=True, eq=False)
class ConcentrationDomain:
    kind: DomainKind
    intervals: Mapping[sp.Symbol, Interval]
    parameter_intervals: Mapping[sp.Symbol, Interval]
    total_constraints: tuple[TotalConstraint, ...] = ()

    @property
    def all_intervals(self):
        return {**self.intervals, **self.parameter_intervals}

    def interval(self, symbol):
        if symbol in self.intervals:
            return self.intervals[symbol]
        try:
            return self.parameter_intervals[symbol]
        except KeyError as error:
            raise KeyError(f"Domain has no symbol '{symbol}'.") from error

    def restrict(self, symbol, *, lower=None, upper=None, strict_lower=False, strict_upper=False):
        current = self.interval(symbol)
        lower = current.lower if lower is None else max(current.lower, exact_expr(lower))
        upper = current.upper if upper is None else min(current.upper, exact_expr(upper))
        lower_closed = (
            current.lower_closed and not strict_lower
            if lower == current.lower
            else not strict_lower
        )
        upper_closed = (
            current.upper_closed and not strict_upper
            if upper == current.upper
            else not strict_upper
        )
        concentrations, parameters = dict(self.intervals), dict(self.parameter_intervals)
        (concentrations if symbol in concentrations else parameters)[symbol] = Interval(
            lower, upper, lower_closed, upper_closed
        )
        return ConcentrationDomain(self.kind, concentrations, parameters, self.total_constraints)

    def is_feasible(self):
        intervals = self.all_intervals
        if any(
            item.lower > item.upper
            or item.lower == item.upper
            and not (item.lower_closed and item.upper_closed)
            for item in intervals.values()
        ):
            return False
        return all(
            sum(intervals[symbol].upper for symbol in total.symbols) > total.minimum
            or sum(intervals[symbol].upper for symbol in total.symbols) == total.minimum
            and all(intervals[symbol].upper_closed for symbol in total.symbols)
            for total in self.total_constraints
        )

    def _satisfy_totals(self, point, costs=None, maximize=False):
        intervals = self.all_intervals
        for total in self.total_constraints:
            remaining = total.minimum - sum(point[symbol] for symbol in total.symbols)
            if remaining <= 0:
                continue
            capacities = {
                symbol: intervals[symbol].upper - point[symbol] for symbol in total.symbols
            }
            if remaining > sum(capacities.values()):
                return False
            if costs is None and remaining < sum(capacities.values()):
                capacity = sum(capacities.values())
                for symbol, available in capacities.items():
                    point[symbol] += remaining * available / capacity
                continue
            order = (
                total.symbols
                if costs is None
                else sorted(
                    total.symbols,
                    key=lambda symbol: (costs.get(symbol, 0), symbol.name),
                    reverse=maximize,
                )
            )
            for symbol in order:
                amount = min(remaining, capacities[symbol])
                if amount and amount == capacities[symbol] and not intervals[symbol].upper_closed:
                    return False
                point[symbol] += amount
                remaining -= amount
                if remaining == 0:
                    break
        return True

    def exact_witness(self, preferences=None):
        if not self.is_feasible():
            return None
        preferences, point = preferences or {}, {}
        for symbol, interval in self.all_intervals.items():
            preferred = preferences.get(symbol)
            point[symbol] = (
                exact_expr(preferred)
                if preferred is not None and interval.contains(preferred)
                else interval.lower
                if interval.lower_closed
                else (interval.lower + interval.upper) / 2
            )
        return point if self._satisfy_totals(point) else None

    def _extreme(self, form, maximize):
        intervals = self.all_intervals
        unknown = set(form.coefficients) - set(intervals)
        if unknown:
            raise ValueError(
                "Affine expression uses symbols outside the domain: "
                + ", ".join(sorted(map(str, unknown)))
                + "."
            )
        point = {}
        for symbol, interval in intervals.items():
            coefficient = form.coefficients.get(symbol, sp.S.Zero)
            if coefficient.is_positive is None and coefficient.is_negative is None:
                raise ValueError(f"Cannot determine affine coefficient sign: {coefficient}.")
            upper = coefficient > 0 if maximize else coefficient < 0
            point[symbol] = interval.upper if upper else interval.lower
        self._satisfy_totals(point, form.coefficients, maximize)
        value = form.value(point)
        witness = self.exact_witness(point)
        return value, witness if witness is not None and form.value(witness) == value else None

    def affine_bounds(self, expression):
        if not self.is_feasible():
            raise ValueError("Cannot bound an expression on an empty domain.")
        form = affine_form(expression)
        if form is None:
            return None
        lower, lower_point = self._extreme(form, False)
        upper, upper_point = self._extreme(form, True)
        return AffineBounds(lower, upper, lower_point, upper_point)


@dataclass(frozen=True)
class DomainSpec:
    symbols: CaseSymbols
    concentration_model: ConcentrationModel
    upper: Mapping[sp.Symbol, sp.Expr]
    excursion_lower: Mapping[sp.Symbol, sp.Expr]
    parameter_intervals: Mapping[sp.Symbol, Interval]
    total_constraints: tuple[TotalConstraint, ...] = ()

    def __post_init__(self):
        model = ConcentrationModel(self.concentration_model)
        upper = {key: exact_expr(value) for key, value in self.upper.items()}
        lower = {key: exact_expr(value) for key, value in self.excursion_lower.items()}
        concentrations = self.symbols.concentration_symbols
        if set(upper) != concentrations or set(lower) != concentrations:
            raise ValueError("Domain bounds must cover every concentration symbol.")
        if any(value <= 0 for value in upper.values()):
            raise ValueError("Concentration upper bounds must be positive.")
        if any(value > 0 for value in lower.values()):
            raise ValueError("Excursion lower bounds must be non-positive.")
        if set(self.parameter_intervals) != self.symbols.parameter_symbols:
            raise ValueError("Domain must bound temperature and pressure.")
        if self.parameter_intervals[self.symbols.temperature].lower <= 0:
            raise ValueError("Temperature lower bound must be positive.")
        constraints, used = tuple(self.total_constraints), set()
        for total in constraints:
            if not set(total.symbols) <= concentrations or used & set(total.symbols):
                raise ValueError("Total constraints must use disjoint concentration groups.")
            used.update(total.symbols)
        if model is ConcentrationModel.INDEPENDENT and constraints:
            raise ValueError("Independent concentration domains cannot define totals.")
        for name, value in (
            ("concentration_model", model),
            ("upper", upper),
            ("excursion_lower", lower),
            ("parameter_intervals", dict(self.parameter_intervals)),
            ("total_constraints", constraints),
        ):
            object.__setattr__(self, name, value)
        for kind in DomainKind:
            if not self.build(kind).is_feasible():
                raise ValueError(f"{kind.value.title()} concentration domain is empty.")

    def build(self, kind):
        kind = DomainKind(kind)
        lower = (
            {symbol: sp.S.Zero for symbol in self.upper}
            if kind is DomainKind.PHYSICAL
            else self.excursion_lower
        )
        return ConcentrationDomain(
            kind,
            {symbol: Interval(lower[symbol], upper) for symbol, upper in self.upper.items()},
            self.parameter_intervals,
            self.total_constraints,
        )
