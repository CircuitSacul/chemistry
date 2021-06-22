from __future__ import annotations

import re
from enum import Enum
from typing import Any, List, Dict, Tuple

from sympy import symbols, Eq, solve, simplify


def multiply(values):
    if len(values) == 0:
        return 1
    v = values.pop()
    for nv in values:
        v = v * nv
    return v


class ItemState(Enum):
    G = "gas"
    S = "solid"
    AQ = "aqueos"
    L = "liquid"
    GRAPHITE = "graphite"

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}.{self.name}"


class Item:  # represents a reactant or product
    def __init__(
        self,
        name: str,
        moles: float = None,
        state: ItemState = None,
        concentration: float = None,
        at_eq: bool = None,
        order: int = None
    ):
        self.moles = moles
        self.state = state
        self.name = name
        self.concentration = concentration
        self.at_eq = at_eq
        self.order = order

    @property
    def pretty_kwargs(self):
        return ", ".join(
            [f"{key}={val!r}" for key, val in self.asdict().items()]
        )

    def asdict(self):
        return {
            "moles": self.moles,
            "state": self.state,
            "name": self.name,
            "concentration": self.concentration,
            "at_eq": self.at_eq,
            "order": self.order,
        }

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = self.pretty_kwargs
        return f"{name}({kwargs})"

    def __str__(self):
        return (
            f"{self.moles}{self.name}"
            f"{'^' + str(self.order) if self.order is not None else ''}"
            f"{'_eq' if self.at_eq else ''}"
            f" ({self.state.name.lower()})"
        )


class Reaction:
    def __init__(
        self,
        reactants: List[Item] = [],
        products: List[Item] = [],
        rate: float = None,
        rate_constant: float = None,
        k: float = None,  # aka equilibrium constant
    ):
        self.reactants = reactants
        self.products = products
        self._rate = rate
        self._rate_constant = rate_constant
        self._k = k

    @property
    def overall_order(self):
        if not all([r.order is not None for r in self.reactants]):
            raise RuntimeError("Not all reactants have an order.")
        return sum([r.order for r in self.reactants])

    @property
    def rate(self):
        if self._rate is not None:
            return self._rate
        if self._rate_constant is None:
            raise RuntimeError("Cannot determine rate without rate constant.")

        k = self.rate_constant
        return k * multiply(
            [r.concentration ** r.order for r in self.reactants]
        )

    @rate.setter
    def rate(self, value):
        self._rate = value

    @property
    def rate_constant(self):
        if self._rate_constant is not None:
            return self._rate_constant
        assert (
            self._rate is not None
        ), "Cannot determine rate constant without rate."
        assert all(
            [r.order is not None for r in self.reactants]
        ), "Not all reactants have an order."

        rate = self.rate

        k = rate / multiply(
            [r.concentration ** r.order for r in self.reactants]
        )
        return k

    @rate_constant.setter
    def rate_constant(self, value):
        self._rate_constant = value

    @property
    def rate_constant_unit(self):
        return f"1/(M^{self.overall_order-1}*s)"

    @property
    def _eq_reactants(self):
        return [
            r
            for r in self.reactants
            if r.state not in [ItemState.S, ItemState.L]
        ]

    @property
    def _eq_products(self):
        return [
            p
            for p in self.products
            if p.state not in [ItemState.S, ItemState.L]
        ]

    @property
    def _at_eq(self):
        return all([i.at_eq for i in self._eq_products + self._eq_reactants])

    @property
    def k(self):
        if self._k is not None:
            return self._k

        assert self._at_eq, "Reaction is not at equilibrium."

        return multiply(
            [p.concentration ** p.moles for p in self._eq_products]
        ) / multiply([r.concentration ** r.moles for r in self._eq_reactants])

    @k.setter
    def k(self, value):
        self._k = value

    @property
    def k_unit(self):
        top_overall = sum([p.moles for p in self._eq_products])
        bot_overall = sum([r.moles for r in self._eq_reactants])
        m = symbols("M")
        return str(simplify(m ** top_overall / m ** bot_overall))

    @property
    def k_equation(self) -> str:
        return "({p}) / ({r})".format(
            p=" * ".join(
                [f"[{p.name}]_eq^{p.moles}" for p in self._eq_products]
            ),
            r=" * ".join(
                [f"[{r.name}]_eq^{r.moles}" for r in self._eq_reactants]
            ),
        )

    def solve_missing_concentrations(self, update_items: bool = True):
        assert self._at_eq, "Reaction not at equilibrium"
        if self._k is None:
            raise RuntimeError(
                "Cannot solve for missing concentrations without knowing "
                "the equilibrium constant beforehand."
            )

        missing = len(
            list(
                filter(
                    lambda i: i.concentration is None,
                    self._eq_products + self._eq_reactants,
                )
            )
        )
        if missing > 1:
            raise RuntimeError(
                "Cannot solve for more than one missing concentration."
            )
        if missing == 0:
            print("Nothing to solve.")
            return

        x = symbols("x")

        solving: Item | None = None
        top = []
        bottom = []
        for product in self._eq_products:
            p = product
            if p.concentration is None:
                p = x
                solving = product
            else:
                p = p.concentration
            p = p ** product.moles
            top.append(p)
        for reactant in self._eq_reactants:
            r = reactant
            if r.concentration is None:
                solving = reactant
                r = x
            else:
                r = r.concentration
            r = r ** reactant.moles
            bottom.append(r)

        result = solve(Eq(multiply(top) / multiply(bottom), self.k))
        assert len(result) == 1, f"Length of result is {len(result)}, not 1."
        result = float(result.pop())

        if update_items:
            solving.concentration = result

        return result

    def __repr__(self):
        return (
            "Reaction(\n"
            "\treactants=[\n"
            + "\n".join([f"\t\t{r!r}," for r in self.reactants])
            + "\n\t],\n"
            "\tproducts=[\n"
            + "\n".join([f"\t\t{p!r}," for p in self.products])
            + "\n\t],\n"
            f"\trate={self._rate},\n"
            f"\trate_constant={self._rate_constant},\n"
            f"\tk={self._k},\n)"
        )

    def __str__(self):
        return (
            " + ".join([str(r) for r in self.reactants])
            + " -> "
            + " + ".join([str(p) for p in self.products])
        )


def determine_orders(trials: List[Reaction]) -> Dict[str, int]:
    o = symbols("o")

    def get_diff_con(r1: Item, r2: Item) -> List[Tuple[int, Item]]:
        return [
            (x, r)
            for x, r in enumerate(r1.reactants)
            if r.concentration != r2.reactants[x].concentration
        ]

    orders: Dict[Item, int] = {}
    for r1 in trials:
        for r2 in trials:
            if r1 is r2:
                continue
            diff = get_diff_con(r1, r2)
            if len(diff) != 1:
                continue
            index, diff = diff.pop()

            left = r1.rate / r2.rate
            right = diff.concentration / r2.reactants[index].concentration
            solutions = solve(Eq(left, right ** o))
            if len(solutions) != 1:
                print(f"Warning: Number of solutions was not 1. ({solutions})")
                continue
            orders[diff.name] = float(solutions[0])

    return orders


def reaction_from_string(
    reaction_text: str,
    **reaction_attrs: Any,
) -> Reaction:
    left, right = reaction_text.split("->")
    left, right = left.strip(), right.strip()

    reactants_text = [r.strip() for r in left.split("+")]
    products_text = [p.strip() for p in right.split("+")]

    def parse_to_items(string_list: list[str]) -> list[Item]:
        items: list[Item] = []
        for text in string_list:
            match = re.search(
                r"(?P<moles>^\d+)?"
                r"(?P<name>[\d\w]+)"
                r"(\^(?P<order>\d+))?"
                r"(?P<eq>_eq)?"
                r" ?\(?(?P<state>[a-zA-Z]+)?\)?",
                text,
            )
            matchd = match.groupdict()
            moles = int(matchd["moles"] or 1)
            name = matchd["name"]
            order = matchd["order"] or None
            at_eq = matchd["eq"] == "_eq"
            state = matchd["state"] or ItemState.G
            if isinstance(state, str):
                state = ItemState.__getitem__(state.upper())

            items.append(
                Item(
                    name=name,
                    moles=moles,
                    state=state,
                    at_eq=at_eq,
                    order=order,
                )
            )
        return items

    reactants = parse_to_items(reactants_text)
    products = parse_to_items(products_text)

    return Reaction(reactants, products, **reaction_attrs)
