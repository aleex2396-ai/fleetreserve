"""Every module under `src/` that reaches the sealed-data gate writes the route's record, so that
`run_evidence` (point 4, reading E3) never reads its results from an earlier commit.

The rule chosen (reading G1 below gives the alternatives). It is conservative on purpose: what
it cannot tell apart it counts as reaching the gate.
1. Units. A module's units are the names it binds at its top level, each with everything that
   binds it: a `def` or a class (its body, methods, decorators, defaults, annotations and base
   classes), an assignment (`x = ...`, `x += ...`, `TABLE[k] = ...` and `obj.attr = ...`
   bind `TABLE` and `obj`, a target of `for` or `with` at the top, a name assigned through
   `global` in a function), an import of a name of this package, and `from m import *` (one
   unit holding every name of `m`); an import of a module is a name for that module, not a
   unit. The module's code that binds no name (`if __name__ == "__main__": ...`, a call at
   the top) is one more unit. A package's `x/__init__.py` is the module `x`, the package's own
   `__init__.py` the module `__init__`.
2. References. A unit refers to a name of a module when anything in it names it: a call, a
   value in a dict, list or tuple, an argument, `functools.partial`, a lambda, a decorator, a
   default, a base class, a return value, an import (also inside a function). Names are resolved
   as Python scopes them (from the standard library's `symtable`): a function's local variable
   is not the module's name of the same spelling. Attribute chains of any length are followed
   through modules and packages (`fleetreserve.heldout_loader.norway_sites`,
   `helper.sub.take_route` where `helper` imported `sub`), with `as` aliases, relative imports
   and `from m import *`. A module taken as a value (`getattr(sealed_route, ...)`,
   `vars(sealed_route)`, `sealed_route.__dict__`, `x = heldout_loader`,
   `importlib.import_module("fleetreserve.sealed_route")`) refers to every name of that module
   and its submodules; an `importlib.import_module` or `__import__` of a
   name built at run time refers to every module whose name begins with its text before the
   first part built at run time (all of them when none is fixed), a variable holding that text
   being read where the same unit assigns it.
3. Reaching the gate. The gate's entries are `heldout_loader.open_sealed` (the opener) and
   `sealed_route.require_gate`. A unit, in any module, that refers to a name that reaches the
   gate reaches it too; the names grow over every module under `src/` (subfolders too) until no
   more are found. So a function holding the gate in a module-level name, a dict of routes, a
   partial, a class whose method passes the gate (and every instance or subclass of it), a
   package re-export, a function that returns a route: each reaches the gate, and so does every
   unit that names it.
4. The record. A unit writes the record when it assigns `[ROUTE_KEY]`, holds `ROUTE_KEY` as a
   key of a dict display, passes `sealed_route=` as a keyword (the key as the name `ROUTE_KEY`,
   an attribute `x.ROUTE_KEY` or its text `"sealed_route"`), or refers to `sealed_route.mark`.
5. What is asked, of each module outside the three that define the gate (`heldout_loader`,
   `sealed_route`, `cold_start_gate`) and outside `NO_RESULTS`:
   (a) a module with a unit that reaches the gate writes the record in some unit of its own (a
       helper module that writes it does not excuse a caller, and a caller that writes it does
       not excuse the helper);
   (b) every entry writes the record in itself or in a unit of the same module it refers to,
       directly or through others of that module. The entries are: each `def` or class that
       reaches the gate and that no other unit of a checked module refers to (`main` below
       `if __name__ == "__main__"` is referred to by the module's code, which is then the
       entry); each unit whose code run at import (outside `def` and class bodies and lambdas)
       calls a name that reaches the gate, a method called on such a name counting as a call of
       it and a class called by its own name counting only when what runs to make one (its body
       outside methods, `__init__`, `__new__`, `__post_init__`, the same of its base classes)
       refers to a name that reaches the gate; and one `def` or class of each group whose
       units are referred to only by each other.
   A module that reaches the gate and writes no results is listed in `NO_RESULTS` with the
   reason, one entry per module; the test also fails for an entry that no longer reaches the
   gate or is not a module. A reference from the three defining modules or from a `NO_RESULTS`
   module does not make a unit stop being an entry, except through an import there that a
   checked module's unit names, which passes the name on (a listed package re-export).

Over-caution (flagged though it may not pass the gate; the remedy is the record or a listed
reason): a class, dict or tuple is one unit, so a module that names only a class's constant,
another key of a route table, or data derived from a route (`tuple(n for n, _ in ROUTES)`)
reaches the gate; a module taken as a value reaches it if any of its names does; a method
called at import on a name that reaches the gate (`ROUTES.values()`) is a call of it; a name
that is local in one function of a unit and global in another counts as global in the whole
unit; a lambda's parameter or a comprehension's variable at the top level is read as the
module's name of that spelling; annotations count as references; a package that re-exports a
name reaching the gate is asked for the record or a listed reason (rule 5a), used or not.

Limits (what is not seen). (b) asks for a record anywhere below an entry, not on every path: a
dispatcher whose one branch writes it excuses the others, and a record on a dead branch, in a
scratch dict or under a wrong value counts; the record's content is not checked. A function
held in a module-level name (alias, lambda, partial, table) that nothing under `src/` names is
not an entry, so it is asked of only through (a); the same holds for a call at import inside a
class body (the class is the unit). A module's attributes other than `__dict__`,
`__getattr__` and `__getattribute__` (`__name__`, `__path__`, `__file__`) are names, not the
module taken as a value. Not followed, being deliberate tricks rather than accidents: a name
built at run time other than an import's text (`getattr(obj, name)` on something not taken from
this package, `globals()`, `sys.modules`, `exec`), import machinery other than the two calls
named (`runpy`, `importlib.util`, `pkgutil` loaders), `dict.update` with a name that is not a
dict display. An object reaching the gate that arrives as a parameter is asked of the unit that
names its class or source, not of the unit that receives it. A command started as a process of
its own (`python -m fleetreserve.x`) is checked as its own module. Modules outside `src/`
(`tools/`, tests) are not scanned. The three modules that define the gate are not checked. This
is a check against the accidents of an honest change, as the other guards of this package are.

Readings (the alternative in brackets):
  G1. The unit of reference is the top-level name, conservative as above, with the record asked
      of each module and of each entry (5a, 5b), because a precise per-call trace is a larger
      scan whose own gaps are harder to state, and every miss the earlier two scans had came
      from a way of naming the gate they did not follow, not from too little precision. On the
      real tree it flags nothing, lists only `gate_run` and `run_driver`, and finds one entry
      per command that reaches the gate: its module's code. [Alternatives: the module as the
      only unit (5a alone), which reads a record written in an unrelated function as covering a
      second command of the same module; a mark left at run time by the gate itself, which no
      scan can miss but which changes the gate's code.]
"""

import ast
import symtable
from pathlib import Path

import pytest

TOP = Path(__file__).resolve().parents[1]
SOURCE = TOP / "src" / "fleetreserve"
PACKAGE = "fleetreserve"
DEFINING = ("heldout_loader", "sealed_route", "cold_start_gate")  # the gate's own modules
SEEDS = {("heldout_loader", "open_sealed"), ("sealed_route", "require_gate")}
KEY = "ROUTE_KEY"
KEY_TEXT = "sealed_route"  # the value of `gate_summary.ROUTE_KEY` (a test below pins it)
MARK = ("sealed_route", "mark")  # `sealed_route.mark` writes the record
TOP_UNIT = "<module>"  # a module's code that binds no name
ANY = "*"  # (module, ANY): the module object itself, taken as a value
MAKING = ("__init__", "__new__", "__post_init__")  # methods run when a class is called

# Modules that reach the gate and record no results file that `read_evidence` could read, each
# with the reason (read in the module named).
NO_RESULTS = {
    "gate_run": (
        "asks the route about the units, and refuses a held-out run before any result is "
        "written (its `main`, after `sealed_route.matrix_route`: `route.held_out` ends in exit 2)"
    ),
    "run_driver": (
        "passes the gate for the sealed run's drive and writes only the drive log "
        "(`runs.noindex/drives/<UTC start>-<purpose>.jsonl`), which no `read_evidence` call reads"
    ),
    "run_list": (
        "the run list's checker: calls `sealed_route.matrix_route` and `units_route` only where "
        "every unit is a development unit or the units mix (`route_fault`, before the folders "
        "and the gate), and writes no results file"
    ),
    "run_starts": (
        "the run list's start check: starts each step of the route in a process of its own in a "
        "made-up repository holding no data, at that repository's made-up tag, stops it at its "
        "first read of data, and writes no results file"
    ),
}

Identity = tuple[str, str]  # (module key, a name it binds at the top, or ANY)


def _join(base: str, name: str) -> str:
    return f"{base}.{name}" if base else name


def _key(dotted: str) -> str:
    """The module key of a dotted path under the package: `x.__init__` is `x`."""
    parts = dotted.split(".")
    return ".".join(parts[:-1]) if parts[-1] == "__init__" else dotted


def _shown(key: str) -> str:
    return key or "__init__"


def _packages(key: str) -> list[str]:
    """The module and every package above it, the package itself (``) last."""
    parts = key.split(".") if key else []
    return [".".join(parts[:n]) for n in range(len(parts), -1, -1)]


class _Unit:
    """One name a module binds at its top (rule 1)."""

    def __init__(self, name: str):
        self.name = name
        self.defined = False  # bound by a `def` or class
        self.module_only = True  # bound only as a name for a module object
        self.imported = False
        self.parts: list[tuple[ast.AST, set[str] | None, dict, bool]] = []
        self.refs: set[Identity] = set()
        self.calls: set[tuple[Identity, bool]] = set()  # called at import time; by its own name
        self.is_class = False
        self.making: list[tuple[ast.AST, set[str] | None, dict]] = []  # what runs to make one
        self.bases: list[ast.AST] = []
        self.making_refs: set[Identity] = set()
        self.base_refs: set[Identity] = set()


class _Module:
    """One module's units and imports, before names are resolved across modules."""

    def __init__(self, dotted: str, text: str, known: set[str]):
        self.package = dotted.split(".")[-1] == "__init__"
        self.key = _key(dotted)
        parts = self.key.split(".") if self.key else []
        self.here = parts if self.package else parts[:-1]
        self.known = known
        self.tree = ast.parse(text)
        self.tables: dict[tuple[str, int], list] = {}
        for child in symtable.symtable(text, dotted, "exec").get_children():
            self.tables.setdefault((child.get_name(), child.get_lineno()), []).append(child)
        self.modules: dict[str, str] = {}  # local name -> module key
        self.stars: list[str] = []
        self.units: dict[str, _Unit] = {}
        self._collect(self.tree.body)

    def bindings(self, node) -> list[tuple[str, tuple]]:
        """(local name, ("module", key) | ("name", key, name) | ("star", key)) for an import of
        this package's modules or names."""
        found = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] != PACKAGE:
                    continue
                if alias.asname:
                    found.append((alias.asname, ("module", ".".join(parts[1:]))))
                else:
                    found.append((PACKAGE, ("module", "")))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                parts = (node.module or "").split(".")
                if parts[0] != PACKAGE:
                    return []
                base = parts[1:]
            else:
                if node.level - 1 > len(self.here):
                    return []
                base = self.here[: len(self.here) - (node.level - 1)]
                base = [*base, *(node.module or "").split(".")]
            base_key = ".".join(part for part in base if part)
            for alias in node.names:
                if alias.name == "*":
                    found.append(("*", ("star", base_key)))
                elif _join(base_key, alias.name) in self.known:
                    found.append(
                        (alias.asname or alias.name, ("module", _join(base_key, alias.name)))
                    )
                else:
                    found.append((alias.asname or alias.name, ("name", base_key, alias.name)))
        return found

    def unit(self, name: str) -> _Unit:
        return self.units.setdefault(name, _Unit(name))

    def _part(self, name: str, node, names=None, local=None, at_import=True) -> None:
        unit = self.unit(name)
        unit.module_only = False
        unit.parts.append((node, names, local or {}, at_import))

    def _globals(self, node) -> set[str] | None:
        """The names a top-level `def` or class takes from the module: global in its scope or a
        scope inside it, or named where Python evaluates them in the module's scope (decorators,
        defaults, annotations, base classes). None when its scope is not found (all count)."""
        tables = self.tables.get((node.name, node.lineno))
        if not tables:
            return None
        names: set[str] = set()
        stack = list(tables)
        while stack:
            table = stack.pop()
            names |= {symbol.get_name() for symbol in table.get_symbols() if symbol.is_global()}
            stack.extend(table.get_children())
        for part in self._header(node):
            names |= {n.id for n in ast.walk(part) if isinstance(n, ast.Name)}
        return names

    @staticmethod
    def _header(node) -> list[ast.AST]:
        """What Python evaluates at the `def` or `class` statement itself, at import time."""
        header: list[ast.AST] = list(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            return header + [*node.bases, *node.keywords]
        arguments = node.args
        header += [*arguments.defaults, *(d for d in arguments.kw_defaults if d)]
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        every += [a for a in (arguments.vararg, arguments.kwarg) if a]
        header += [a.annotation for a in every if a.annotation]
        return header + ([node.returns] if node.returns else [])

    def _local_imports(self, node) -> dict:
        found: dict = {}
        for item in ast.walk(node):
            if isinstance(item, ast.Import | ast.ImportFrom):
                for local, target in self.bindings(item):
                    if target[0] == "star":
                        found.setdefault("*", []).append(target[1])
                    else:
                        found[local] = target
        return found

    def _targets(self, target) -> list[str]:
        """The module's names that an assignment to `target` binds or changes."""
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, ast.Tuple | ast.List):
            return [name for item in target.elts for name in self._targets(item)]
        if isinstance(target, ast.Starred):
            return self._targets(target.value)
        while isinstance(target, ast.Attribute | ast.Subscript):
            target = target.value
        if isinstance(target, ast.Name) and target.id not in self.modules:
            return [target.id]  # `TABLE[k] = f` and `obj.attr = f`: TABLE and obj hold f
        return [TOP_UNIT]  # `module.name = f`: code run at import

    def _stores(self, node) -> list[str]:
        """Names a statement at the top binds besides its targets (`for x in`, `with ... as x`,
        `x := ...`), not looking inside lambdas and comprehensions (their own scopes)."""
        found = []
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, ast.Lambda | ast.ListComp | ast.SetComp | ast.DictComp):
                continue
            if isinstance(item, ast.GeneratorExp):
                continue
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                found.append(item.id)
            stack.extend(ast.iter_child_nodes(item))
        return found

    def _collect(self, body) -> None:
        for statement in body:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names = self._globals(statement)
                local = self._local_imports(statement)
                self._part(statement.name, statement, names, local, at_import=False)
                for part in self._header(statement):
                    self._part(statement.name, part)
                self.unit(statement.name).defined = True
                if isinstance(statement, ast.ClassDef):
                    making = [
                        item
                        for item in statement.body
                        if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                        or item.name in MAKING
                    ]
                    self.unit(statement.name).is_class = True
                    self.unit(statement.name).making += [(m, names, local) for m in making]
                    self.unit(statement.name).bases += statement.bases
                for item in ast.walk(statement):
                    if isinstance(item, ast.Global):  # `global X` then `X = f` inside it
                        for name in item.names:
                            self.unit(name).refs.add((self.key, statement.name))
                            self.unit(name).module_only = False
            elif isinstance(statement, ast.Import | ast.ImportFrom):
                for local, target in self.bindings(statement):
                    if target[0] == "star":  # `from m import *` brings in every name of m
                        self.stars.append(target[1])
                        unit = self.unit("*")
                        unit.module_only = False
                        unit.imported = True
                        unit.refs.add((target[1], ANY))
                    elif target[0] == "module":
                        self.modules[local] = target[1]
                        self.unit(local).refs.add((target[1], ANY))
                    else:
                        unit = self.unit(local)
                        unit.module_only = False
                        unit.imported = True
                        unit.refs.add((target[1], target[2]))
            elif isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = getattr(statement, "targets", None) or [statement.target]
                bound = {name for target in targets for name in self._targets(target)}
                for name in bound | set(self._stores(statement)):
                    self._part(name, statement)
            else:
                header, bodies = [], []
                for value in (v for _, v in ast.iter_fields(statement)):
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        if isinstance(item, ast.stmt):
                            bodies.append(item)
                        elif isinstance(item, ast.ExceptHandler):
                            header.append(item.type)
                            bodies += item.body
                            if item.name:
                                self.unit(item.name).module_only = False
                        elif isinstance(item, ast.match_case):
                            header += [item.pattern, item.guard]
                            bodies += item.body
                        elif isinstance(item, ast.AST):
                            header.append(item)
                header = [part for part in header if part is not None]
                stored = {name for part in header for name in self._stores(part)}
                for part in header:  # `for x in TABLE`: x holds what TABLE holds
                    for name in [TOP_UNIT, *sorted(stored)]:
                        self._part(name, part)
                self._collect(bodies)


class Scan:
    """The parsed modules of one source tree and the names that reach the gate (rules 1 to 3)."""

    def __init__(self, sources: dict[str, str]):
        known = {_key(dotted) for dotted in sources}
        self.modules: dict[str, _Module] = {}
        for dotted, text in sources.items():
            module = _Module(dotted, text, known)
            self.modules[module.key] = module
        self.trees = {key: module.tree for key, module in self.modules.items()}
        for module in self.modules.values():
            for unit in module.units.values():
                for node, names, local, at_import in unit.parts:
                    refs, calls = self._references(module, node, names, local, at_import)
                    unit.refs |= refs
                    unit.calls |= calls
                for node, names, local in unit.making:
                    unit.making_refs |= self._references(module, node, names, local, False)[0]
                for node in unit.bases:
                    unit.base_refs |= self._references(module, node, None, {}, False)[0]
                unit.refs = {self._bound(ref) for ref in unit.refs}
                unit.calls = {(self._bound(ref), own) for ref, own in unit.calls}
                unit.making_refs = {self._bound(ref) for ref in unit.making_refs}
                unit.base_refs = {self._bound(ref) for ref in unit.base_refs}
        self.reaching = self._grow()

    def _bound(self, ref: Identity) -> Identity:
        """An imported name where it is bound: `(m, n)` for a name `m` has from a star import
        is the name of the module that binds it."""
        if ref[1] == ANY or ref[0] not in self.modules:
            return ref
        found = self._attribute(ref[0], ref[1])
        return (found[1], ANY) if found[0] == "module" else found[1]

    # --- resolving names ----------------------------------------------------------------------
    def _name(self, module: _Module, name: str, names, local) -> tuple | None:
        if name in local:
            target = local[name]
            return ("module", target[1]) if target[0] == "module" else ("id", target[1:])
        if names is not None and name not in names:
            return None  # a local variable of a function
        return self._in(module.key, name, local.get("*", []))

    def _in(self, key: str, name: str, stars=(), seen=frozenset()) -> tuple | None:
        module = self.modules.get(key)
        if module is None:
            return None
        if name in module.modules:
            return ("module", module.modules[name])
        if name in module.units:
            return ("id", (key, name))
        for star in [*stars, *module.stars]:
            if star not in seen | {key}:
                found = self._in(star, name, (), seen | {key})
                if found:
                    return found
        return None

    def _attribute(self, key: str, attr: str) -> tuple:
        if _join(key, attr) in self.modules:
            return ("module", _join(key, attr))
        module = self.modules.get(key)
        if module is not None and attr in module.modules:
            return ("module", module.modules[attr])
        if module is None or attr in module.units:
            return ("id", (key, attr))
        if attr in ("__dict__", "__getattr__", "__getattribute__"):  # the module as a value
            return ("module", key)
        return self._in(key, attr) or ("id", (key, attr))  # a star import's name, where bound

    def _resolve(self, module: _Module, item, names, local) -> Identity | None:
        """The identity a name or attribute chain stands for, a module as (key, ANY)."""
        chain = []
        while isinstance(item, ast.Attribute):
            chain.append(item.attr)
            item = item.value
        if not isinstance(item, ast.Name):
            return None
        target = self._name(module, item.id, names, local)
        for attr in reversed(chain):
            if target is None or target[0] != "module":
                break
            target = self._attribute(target[1], attr)
        if target is None:
            return None
        return (target[1], ANY) if target[0] == "module" else target[1]

    def _imported(self, call: ast.Call, within) -> set[Identity]:
        """`import_module(...)` or `__import__(...)` naming this package's modules (rule 2)."""
        func = call.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called not in ("import_module", "__import__") or not call.args:
            return set()
        texts = [call.args[0]]
        if isinstance(texts[0], ast.Name):  # a variable the same unit assigns its text
            given = [
                item.value
                for item in ast.walk(within)
                if isinstance(item, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == texts[0].id for t in item.targets)
            ]
            if given and all(isinstance(v, ast.Constant | ast.JoinedStr) for v in given):
                texts = given
        found: set[Identity] = set()
        for text in texts:
            if isinstance(text, ast.Constant) and isinstance(text.value, str):
                whole, prefix = True, text.value
            elif isinstance(text, ast.JoinedStr):
                whole, prefix = False, ""
                for value in text.values:
                    if not isinstance(value, ast.Constant):
                        break
                    prefix += str(value.value)
            else:
                return {("", ANY)}
            if whole and prefix == PACKAGE:
                found.add(("", ANY))
            elif whole and prefix.startswith(PACKAGE + "."):
                found.add((prefix[len(PACKAGE) + 1 :], ANY))
            elif not whole and prefix.startswith(PACKAGE + "."):
                rest = prefix[len(PACKAGE) + 1 :]
                found |= {(key, ANY) for key in self.modules if key.startswith(rest)}
                found |= {("", ANY)} if not rest else set()
            elif not whole and PACKAGE.startswith(prefix) or prefix.startswith("."):
                found.add(("", ANY))
        return found

    def _references(self, module, node, names, local, at_import) -> tuple[set, set]:
        """What `node` refers to (rule 2), and the names it calls when run at import time."""
        refs: set[Identity] = set()
        calls: set[tuple[Identity, bool]] = set()
        stack = [(node, at_import)]
        while stack:
            item, running = stack.pop()
            if isinstance(item, ast.Attribute | ast.Name):
                found = self._resolve(module, item, names, local)
                if found is not None:
                    refs.add(found)
                    continue
                if isinstance(item, ast.Name):
                    continue
                while isinstance(item, ast.Attribute):
                    item = item.value
                stack.append((item, running))
                continue
            if isinstance(item, ast.Call):
                refs |= self._imported(item, node)
                if running:
                    func = item.func
                    while isinstance(func, ast.Call | ast.Subscript | ast.Attribute):
                        found = self._resolve(module, func, names, local)
                        if found is not None:
                            break
                        func = func.func if isinstance(func, ast.Call) else func.value
                    found = self._resolve(module, func, names, local)
                    if found is not None:  # by its own name: `Runner()`, not `Runner.open()`
                        last = getattr(func, "attr", getattr(func, "id", None))
                        calls.add((found, func is item.func and last == found[1]))
            inside = running and not isinstance(
                item, ast.Lambda | ast.FunctionDef | ast.AsyncFunctionDef
            )
            stack.extend((child, inside) for child in ast.iter_child_nodes(item))
        return refs, calls

    # --- the names that reach the gate (rule 3) -----------------------------------------------
    def held(self, ref: Identity, names: set[Identity], packages: set[str]) -> bool:
        return ref in names or (ref[1] == ANY and ref[0] in packages)

    def _grow(self) -> set[Identity]:
        names = set(SEEDS)
        self.packages = {key for seed in SEEDS for key in _packages(seed[0])}
        grown = True
        while grown:
            grown = False
            for key, module in self.modules.items():
                for unit in module.units.values():
                    if (key, unit.name) in names:
                        continue
                    if any(self.held(ref, names, self.packages) for ref in unit.refs):
                        names.add((key, unit.name))
                        self.packages |= set(_packages(key))
                        grown = True
        return names

    def reaches(self, ref: Identity) -> bool:
        return self.held(ref, self.reaching, self.packages)

    def runs_the_gate(self, ref: Identity, own_name: bool, seen=frozenset()) -> bool:
        """Whether calling `ref` may pass the gate: for a class called by its own name, whether
        what runs to make one (its body outside methods, `__init__`, `__new__`,
        `__post_init__`, and the same of its base classes) refers to a name that reaches it."""
        module = self.modules.get(ref[0])
        unit = module.units.get(ref[1]) if module else None
        if not own_name or unit is None or ref in seen:
            return self.reaches(ref)
        if unit.imported:
            return any(self.runs_the_gate(r, True, seen | {ref}) for r in unit.refs)
        if not unit.is_class:
            return self.reaches(ref)
        return any(self.reaches(r) for r in unit.making_refs) or any(
            self.runs_the_gate(r, True, seen | {ref}) for r in unit.base_refs
        )

    def units(self, key: str) -> list[_Unit]:
        """The module's units, without the names that stand only for a module object."""
        return [unit for unit in self.modules[key].units.values() if not unit.module_only]

    def reaches_the_gate(self, key: str) -> bool:
        return any((key, unit.name) in self.reaching for unit in self.units(key))


def _writes(node) -> bool:
    """Rule 4 in `node`, `sealed_route.mark` aside."""

    def is_key(item) -> bool:
        return any(
            (isinstance(n, ast.Name) and n.id == KEY)
            or (isinstance(n, ast.Attribute) and n.attr == KEY)
            or (isinstance(n, ast.Constant) and n.value == KEY_TEXT)
            for n in ast.walk(item)
        )

    for item in ast.walk(node):
        stored = isinstance(item, ast.Subscript) and isinstance(item.ctx, ast.Store)
        if stored and is_key(item.slice):
            return True
        if isinstance(item, ast.Dict) and any(k is not None and is_key(k) for k in item.keys):
            return True
        if isinstance(item, ast.keyword) and item.arg == KEY_TEXT:
            return True
    return False


def _records(unit: _Unit) -> bool:
    return MARK in unit.refs or any(_writes(node) for node, _, _, _ in unit.parts)


def entries(scan: Scan, checked: set[str]) -> list[Identity]:
    """The entries of the checked modules (rule 5b)."""
    referrers: dict[Identity, set[Identity]] = {}
    for key in checked:
        for unit in scan.modules[key].units.values():
            for ref in unit.refs - {(key, unit.name)}:
                referrers.setdefault(ref, set()).add((key, unit.name))
    passed_on = True  # an import that a checked module's unit names passes its name on
    while passed_on:
        passed_on = False
        for key, module in scan.modules.items():
            for unit in module.units.values():
                if unit.imported and key not in checked and (key, unit.name) in referrers:
                    for ref in unit.refs:
                        if (key, unit.name) not in referrers.setdefault(ref, set()):
                            referrers[ref].add((key, unit.name))
                            passed_on = True
    reaching_defs = sorted(
        (key, unit.name)
        for key in checked
        for unit in scan.units(key)
        if unit.defined and (key, unit.name) in scan.reaching
    )
    found = sorted(
        {identity for identity in reaching_defs if identity not in referrers}
        | {
            (key, unit.name)
            for key in checked
            for unit in scan.units(key)
            if any(scan.runs_the_gate(ref, own_name) for ref, own_name in unit.calls)
        }
    )
    grouped: set[Identity] = set()
    for identity in reaching_defs:  # one `def` of each group that refers only to each other
        if identity in referrers and identity not in grouped:
            above: set[Identity] = set()
            stack = list(referrers[identity])
            while stack:
                item = stack.pop()
                if item not in above:
                    above.add(item)
                    stack.extend(referrers.get(item, ()))
            if above <= below(scan, identity, checked):
                found.append(identity)
                grouped |= above
    return found


def below(scan: Scan, start: Identity, checked: set[str], inside: str | None = None) -> set:
    """The units of checked modules (of the module `inside` alone, when given) that `start`
    refers to, directly or through others, `start` included."""
    seen: set[Identity] = set()
    stack = [start]
    while stack:
        key, name = stack.pop()
        module = scan.modules.get(key)
        if (key, name) in seen or module is None or name not in module.units:
            continue
        if key not in checked and not module.units[name].imported:
            continue  # an import of an unchecked module (a listed re-export) passes on
        if inside is not None and key != inside:
            continue
        seen.add((key, name))
        stack.extend(module.units[name].refs)
    return seen


def unrecognised(sources: dict[str, str], no_results=NO_RESULTS) -> list[str]:
    """The modules (outside the three defining ones) that reach the gate and do not meet rule
    5, and the entries of `no_results` that do not reach the gate or are not modules, as text;
    empty when every run through the gate is recognised."""
    scan = Scan(sources)
    checked = {key for key in scan.modules if key not in DEFINING and _shown(key) not in no_results}
    starts = entries(scan, checked)
    problems = []
    for key in sorted(scan.modules):
        if key in DEFINING:
            continue
        reaches = scan.reaches_the_gate(key)
        if _shown(key) in no_results:
            if not reaches:
                problems.append(f"{_shown(key)}: listed in NO_RESULTS but does not reach the gate")
            continue
        if not reaches:
            continue
        if not any(_records(unit) for unit in scan.units(key)):
            problems.append(f"{_shown(key)}: reaches the gate and writes no {KEY} record")
            continue
        module = scan.modules[key]
        for _, name in (entry for entry in starts if entry[0] == key):
            if not any(
                _records(module.units[n]) for _, n in below(scan, (key, name), checked, key)
            ):
                problems.append(
                    f"{_shown(key)}: {name} reaches the gate and leads to no {KEY} record"
                )
    shown = {_shown(key) for key in scan.modules}
    problems += [
        f"{key}: listed in NO_RESULTS but not a module" for key in no_results if key not in shown
    ]
    return problems


def sources_of(folder: Path) -> dict[str, str]:
    """Every `.py` file under `folder`, subfolders too, by its dotted path from `folder`."""
    return {
        ".".join(path.relative_to(folder).with_suffix("").parts): path.read_text(encoding="utf-8")
        for path in sorted(folder.rglob("*.py"))
    }


def test_every_module_that_reaches_the_gate_writes_the_routes_record():
    assert unrecognised(sources_of(SOURCE)) == []


def test_the_key_text_is_the_route_key():
    from fleetreserve.gate_summary import ROUTE_KEY

    assert ROUTE_KEY == KEY_TEXT


def test_the_scan_finds_the_gates_entries_and_the_modules_that_reach_it():
    modules = Scan(sources_of(SOURCE))
    assert (
        SEEDS
        | {
            ("heldout_loader", "norway_sites"),
            ("heldout_loader", "asr"),
            ("heldout_loader", "korea"),
            ("sealed_route", "matrix_route"),
            ("sealed_route", "units_route"),
            ("cold_start_gate", "require_gate"),
            ("drift_run", "take_route"),
            ("o3_run", "take_route"),
            ("ceiling_report", "take_route"),
        }
        <= modules.reaching
    )
    callers = {k for k in modules.trees if k not in DEFINING and modules.reaches_the_gate(k)}
    # the seven commands of the route, the cold start, the held-out truth build, and the three
    # of NO_RESULTS, and no other module
    assert callers == {
        "learn_run",
        "compare_run",
        "corner_run",
        "drift_run",
        "o3_run",
        "ceiling_report",
        "ledger_run",
        "cold_start",
        "heldout_truth_build",
        *NO_RESULTS,
    }


# --- made-up modules: the scan goes red for a new module that reaches the gate unrecorded ------

DEFINERS = {
    "heldout_loader": "def open_sealed(root):\n    return b''\ndef norway_sites(root):\n"
    "    return open_sealed(root)\n",
    "sealed_route": "def require_gate(root):\n    return 'tag'\ndef _gate(root):\n"
    "    return require_gate(root)\ndef units_route(root):\n    return _gate(root)\n"
    "def gate_again(route, root):\n    return _gate(root)\n"
    "def mark(document, route):\n    document['sealed_route'] = route\n",
    "cold_start_gate": "from fleetreserve.sealed_route import require_gate\n",
    # two commands that wrap the gate in a function, as `drift_run` and `o3_run` do
    "drift_run": "from fleetreserve import sealed_route\n"
    "def take_route(root, args):\n    return sealed_route.units_route(root)\n"
    "def main(root):\n    document = {}\n    document[ROUTE_KEY] = take_route(root, 1)\n",
    "o3_run": "from fleetreserve import sealed_route\n"
    "def take_route(root, args):\n    return sealed_route.units_route(root)\n"
    "def main(root):\n    document = {}\n    document[ROUTE_KEY] = take_route(root, 1)\n",
}
CALLER = "from fleetreserve import sealed_route\ndef main(root):\n    sealed_route.{call}(root)\n"
WRITES_KEY = "    document = {}\n    document[ROUTE_KEY] = {}\n"
MARKS = "    sealed_route.mark({}, 1)\n"


def _with(**modules) -> dict[str, str]:
    return {**DEFINERS, **modules}


def found(**modules) -> list[str]:
    return unrecognised(_with(**modules), {})


def named(*keys: str) -> list[str]:
    return [f"{key}: reaches the gate and writes no ROUTE_KEY record" for key in keys]


# what the first version found (a call in a module of its own, direct)
@pytest.mark.parametrize("call", ["require_gate", "units_route", "_gate"])
def test_a_new_caller_without_the_record_is_found(call):
    assert found(new_command=CALLER.format(call=call)) == named("new_command")


def test_a_new_route_function_that_calls_the_gate_is_followed():
    sealed = DEFINERS["sealed_route"] + "def another_route(root):\n    return _gate(root)\n"
    assert found(sealed_route=sealed, new_command=CALLER.format(call="another_route")) == named(
        "new_command"
    )


def test_a_new_caller_that_reads_the_opener_without_the_record_is_found():
    source = "from fleetreserve import heldout_loader\nR = (heldout_loader.norway_sites,)\n"
    assert found(new_build=source) == named("new_build")


@pytest.mark.parametrize(
    "source",
    [
        "from fleetreserve.heldout_loader import open_sealed as z\nz(1)\n",
        "from . import heldout_loader\nheldout_loader.open_sealed(1)\n",
        "from fleetreserve.sealed_route import require_gate\n",
        "from fleetreserve import sealed_route\nsealed_route.gate_again(1)\n",
        "from fleetreserve import sealed_route\nsealed_route.units_route(1)\n"
        "d = {}\nd[NOT_ROUTE_KEY] = 1\n",
    ],
)
def test_the_first_versions_finds_stay_found(source):
    assert found(new_command=source) == named("new_command")


@pytest.mark.parametrize("ending", [WRITES_KEY, MARKS])
def test_a_new_caller_with_the_record_is_recognised(ending):
    assert found(new_command=CALLER.format(call="require_gate") + ending) == []


def test_reading_the_key_is_not_writing_it():
    reads = CALLER.format(call="require_gate") + "    return document.get(ROUTE_KEY)\n"
    assert found(new_command=reads) == named("new_command")


def test_a_local_name_that_is_a_gate_name_is_not_the_gate():
    assert found(other="def main(asr):\n    return asr\n", also="asr = 1\n") == []


def test_a_module_that_does_not_reach_the_gate_is_not_asked_for_the_record():
    assert found(other="def main():\n    return 1\n") == []


def test_a_listed_module_that_no_longer_reaches_the_gate_is_found():
    sources = _with(old="def main():\n    return 1\n")
    assert unrecognised(sources, {"old": "reason"}) == [
        "old: listed in NO_RESULTS but does not reach the gate"
    ]
    assert unrecognised(sources, {"gone": "reason"}) == [
        "gone: listed in NO_RESULTS but not a module"
    ]


# (a) a wrapper in another module, called as an attribute, by import, by an aliased import
@pytest.mark.parametrize(
    "source",
    [
        "from fleetreserve import drift_run\ndef main(root, args):\n"
        "    return drift_run.take_route(root, args)\n",
        "from fleetreserve import o3_run\ndef main(root, args):\n"
        "    return o3_run.take_route(root, args)\n",
        "from fleetreserve.drift_run import take_route\ndef main(root, args):\n"
        "    return take_route(root, args)\n",
        "from fleetreserve.drift_run import take_route as route\ndef main(root, args):\n"
        "    return route(root, args)\n",
        "from fleetreserve import drift_run as d\ndef main(root, args):\n"
        "    return d.take_route(root, args)\n",
        "from . import drift_run\ndef main(root, args):\n    return drift_run.take_route(1, 2)\n",
    ],
)
def test_a_module_that_reaches_the_gate_through_a_wrapper_in_another_module_is_found(source):
    assert found(new_command=source) == named("new_command")


def test_a_wrapper_of_a_wrapper_is_followed():
    middle = (
        "from fleetreserve.drift_run import take_route\n"
        "def route_of(root, args):\n    return take_route(root, args)\n"
        "def write(document):\n    document[ROUTE_KEY] = 1\n"
    )
    outer = (
        "from fleetreserve import middle\ndef main(root, args):\n    middle.route_of(root, args)\n"
    )
    assert found(middle=middle, outer=outer) == named("outer")


# (b) aliases of the gate's own modules
@pytest.mark.parametrize(
    "source",
    [
        "from fleetreserve import heldout_loader as loader\nR = loader.norway_sites\n",
        "from fleetreserve import cold_start_gate as g\ndef main(root):\n"
        "    g.require_gate(root)\n",
        "from fleetreserve import sealed_route as route\ndef main(root):\n"
        "    route.units_route(root)\n",
        "import fleetreserve.heldout_loader as loader\nR = loader.norway_sites\n",
        "from fleetreserve.sealed_route import units_route as u\ndef main(root):\n    u(root)\n",
    ],
)
def test_a_module_that_reaches_the_gate_through_an_alias_is_found(source):
    assert found(new_command=source) == named("new_command")


def test_an_alias_that_writes_the_record_is_recognised():
    source = (
        "from fleetreserve import sealed_route as route\n"
        "from fleetreserve import gate_summary as gs\n"
        "def main(root):\n    document = {}\n    route.units_route(root)\n"
        "    document[gs.ROUTE_KEY] = 1\n"
    )
    assert found(new_command=source, gate_summary="ROUTE_KEY = 'sealed_route'\n") == []


# (c) a helper module that writes the record in one function; its callers are still followed
HELPER = (
    "from fleetreserve import sealed_route\n"
    "def route(root):\n    return sealed_route.units_route(root)\n"
    "def record(document):\n    document[ROUTE_KEY] = 1\n"
)


def test_a_helper_that_writes_the_record_does_not_excuse_its_callers():
    caller = "from fleetreserve import helper\ndef main(root):\n    return helper.route(root)\n"
    assert found(helper=HELPER, command=caller) == named("command")


def test_a_caller_of_the_helper_that_writes_the_record_itself_is_recognised():
    caller = (
        "from fleetreserve import helper\ndef main(root):\n    document = {}\n"
        "    helper.route(root)\n    document[ROUTE_KEY] = 1\n"
    )
    assert found(helper=HELPER, command=caller) == []


# (d) a module in a subfolder
def test_a_module_in_a_subfolder_is_found(tmp_path):
    (tmp_path / "models").mkdir()
    for key, text in DEFINERS.items():
        (tmp_path / f"{key}.py").write_text(text)
    (tmp_path / "models" / "new_model.py").write_text(CALLER.format(call="require_gate"))
    sources = sources_of(tmp_path)
    assert "models.new_model" in sources
    assert unrecognised(sources, {}) == named("models.new_model")


def test_a_relative_import_in_a_subfolder_is_resolved():
    source = "from .. import sealed_route\ndef main(root):\n    sealed_route.require_gate(root)\n"
    assert found(**{"models.new_model": source}) == named("models.new_model")
    sibling = "from . import helper\ndef main(root):\n    helper.route(root)\n"
    assert found(**{"models.helper": HELPER, "models.new_model": sibling}) == named(
        "models.new_model"
    )


# --- forms an honest change could write: each is found, or is stated in the limits ------------
# A helper module below writes the record in a function of its own (`record`), so that rule 5a
# holds for it and the case tests whether its caller is followed.

IMPORT = "from fleetreserve import sealed_route\n"
RECORD = "def record(document):\n    document[ROUTE_KEY] = 1\n"
RUNNER = (
    "class Runner:\n    LABEL = 'x'\n    def open(self, r):\n"
    "        return sealed_route.units_route(r)\n"
)
USES_HELPER = "from fleetreserve import helper\ndef main(r):\n    return {call}\n"


def helper(body: str) -> str:
    return IMPORT + body + RECORD


def uses(call: str) -> str:
    return USES_HELPER.format(call=call)


def entry(key: str, name: str) -> list[str]:
    return [f"{key}: {name} reaches the gate and leads to no ROUTE_KEY record"]


FORMS = {
    # the second review's finding 1: a gate function held in a module-level name
    "alias in a module-level name": (
        {
            "helper": helper("units = sealed_route.units_route\n"),
            "command": uses("helper.units(r)"),
        },
        named("command"),
    ),
    "dict of routes, as an attribute": (
        {
            "helper": helper("ROUTES = {'units': sealed_route.units_route}\n"),
            "command": uses("helper.ROUTES['units'](r)"),
        },
        named("command"),
    ),
    "dict of routes, imported by name": (
        {
            "helper": helper("ROUTES = {'units': sealed_route.units_route}\n"),
            "command": "from fleetreserve.helper import ROUTES\ndef main(r):\n"
            "    return ROUTES['units'](r)\n",
        },
        named("command"),
    ),
    "functools.partial": (
        {
            "helper": helper(
                "import functools\nunits = functools.partial(sealed_route.units_route)\n"
            ),
            "command": uses("helper.units(r)"),
        },
        named("command"),
    ),
    "lambda in a module-level name": (
        {
            "helper": helper("go = lambda r: sealed_route.units_route(r)\n"),
            "command": uses("helper.go(r)"),
        },
        named("command"),
    ),
    "name bound through `global` in a function": (
        {
            "helper": helper(
                "ROUTE = None\ndef setup():\n    global ROUTE\n"
                "    ROUTE = sealed_route.units_route\n"
            ),
            "command": uses("helper.ROUTE(r)"),
        },
        named("command"),
    ),
    # the second review's finding 3: a method on an object of a class from another module
    "method, class imported by name": (
        {
            "helper": helper(RUNNER),
            "command": "from fleetreserve.helper import Runner\ndef main(r):\n"
            "    return Runner().open(r)\n",
        },
        named("command"),
    ),
    "method, class as an attribute": (
        {"helper": helper(RUNNER), "command": uses("helper.Runner().open(r)")},
        named("command"),
    ),
    "method on the class": (
        {
            "helper": helper(RUNNER),
            "command": "from fleetreserve.helper import Runner\ndef main(r):\n"
            "    return Runner.open(None, r)\n",
        },
        named("command"),
    ),
    "method of a subclass": (
        {
            "helper": helper(RUNNER),
            "command": "from fleetreserve import helper\nclass Mine(helper.Runner):\n    pass\n"
            "def main(r):\n    return Mine().open(r)\n",
        },
        named("command"),
    ),
    "method of a module-level instance": (
        {
            "helper": helper(RUNNER + "RUNNER = Runner()\n"),
            "command": uses("helper.RUNNER.open(r)"),
        },
        named("command"),
    ),
    "instance made here, used in another module": (
        {
            "helper": helper(RUNNER),
            "work": "def run(runner, r):\n    return runner.open(r)\n",
            "command": "from fleetreserve import helper, work\ndef main(r):\n"
            "    return work.run(helper.Runner(), r)\n",
        },
        named("command"),
    ),
    # functions that hand the gate on
    "route returned by a function": (
        {
            "helper": helper("def pick():\n    return sealed_route.units_route\n"),
            "command": uses("helper.pick()(r)"),
        },
        named("command"),
    ),
    "decorator": (
        {
            "helper": helper(
                "def sealed(fn):\n    def run(r):\n        sealed_route.require_gate(r)\n"
                "        return fn(r)\n    return run\n"
            ),
            "command": "from fleetreserve import helper\n@helper.sealed\ndef main(r):\n"
            "    return 1\n",
        },
        named("command"),
    ),
    "default argument": (
        {
            "helper": helper("def run(r, route=sealed_route.units_route):\n    return route(r)\n"),
            "command": uses("helper.run(r)"),
        },
        named("command"),
    ),
    "gate function passed as an argument": (
        {
            "work": "def call(f, r):\n    return f(r)\n",
            "command": "from fleetreserve import sealed_route, work\ndef main(r):\n"
            "    return work.call(sealed_route.units_route, r)\n",
        },
        named("command"),
    ),
    "registry filled by a decorator": (
        {
            "registry": "COMMANDS = {}\ndef command(fn):\n    COMMANDS[fn.__name__] = fn\n"
            "    return fn\n",
            "dispatch": "from fleetreserve import registry\ndef main(name):\n"
            "    return registry.COMMANDS[name](1)\n",
            "cmd": "from fleetreserve import registry, sealed_route\n@registry.command\n"
            "def run(r):\n    sealed_route.units_route(r)\n",
        },
        named("cmd"),
    ),
    # imports
    "import inside a function": (
        {
            "command": "def main(r):\n    from fleetreserve import sealed_route\n"
            "    sealed_route.units_route(r)\n"
        },
        named("command"),
    ),
    "dotted import without `as`": (
        {
            "command": "import fleetreserve.heldout_loader\ndef main(r):\n"
            "    fleetreserve.heldout_loader.norway_sites(r)\n"
        },
        named("command"),
    ),
    "getattr with a literal name": (
        {"command": IMPORT + "def main(r):\n    getattr(sealed_route, 'units_route')(r)\n"},
        named("command"),
    ),
    "importlib with a literal name": (
        {
            "command": "import importlib\ndef main(r):\n"
            "    importlib.import_module('fleetreserve.sealed_route').units_route(r)\n"
        },
        named("command"),
    ),
    "importlib with a name built at run time": (
        {
            "command": "import importlib\ndef main(name, r):\n"
            "    importlib.import_module(f'fleetreserve.{name}').take_route(r, 1)\n"
        },
        named("command"),
    ),
    "star import": (
        {"command": "from fleetreserve.sealed_route import *\ndef main(r):\n    units_route(r)\n"},
        named("command"),
    ),
    "module object in a module-level name": (
        {
            "command": "from fleetreserve import heldout_loader\nloader = heldout_loader\n"
            "def main(r):\n    loader.norway_sites(r)\n"
        },
        named("command"),
    ),
    "module reached through another module's import": (
        {
            "helper": "from fleetreserve import drift_run as sub\n",
            "command": uses("helper.sub.take_route(r, 1)"),
        },
        named("command"),
    ),
    "module imported from another module by name": (
        {
            "helper": "from fleetreserve import drift_run as sub\n",
            "command": "from fleetreserve.helper import sub\ndef main(r):\n"
            "    sub.take_route(r, 1)\n",
        },
        named("command"),
    ),
    "relative import of a wrapper from a subfolder": (
        {"models.new": "from ..drift_run import take_route\ndef main(r):\n    take_route(r, 1)\n"},
        named("models.new"),
    ),
    "import tried, def under `if`": (
        {
            "command": "try:\n    from fleetreserve.drift_run import take_route\n"
            "except ImportError:\n    take_route = None\nif True:\n"
            "    def main(r):\n        take_route(r, 1)\n"
        },
        named("command"),
    ),
    "async def": (
        {"command": IMPORT + "async def main(r):\n    sealed_route.units_route(r)\n"},
        named("command"),
    ),
    "name from a star import, imported again by name": (
        {
            "helper": "from fleetreserve.sealed_route import *\n",
            "command": "from fleetreserve.helper import units_route\ndef main(r):\n"
            "    units_route(r)\n",
        },
        named("command", "helper"),
    ),
    "the module's own dictionary": (
        {"command": IMPORT + "def main(r):\n    sealed_route.__dict__['units_route'](r)\n"},
        named("command"),
    ),
    "vars of the module": (
        {"command": IMPORT + "def main(r):\n    vars(sealed_route)['units_route'](r)\n"},
        named("command"),
    ),
    "closure over a local name": (
        {
            "helper": helper(
                "def outer():\n    route = sealed_route.units_route\n"
                "    def inner(r):\n        return route(r)\n    return inner\n"
            ),
            "command": uses("helper.outer()(r)"),
        },
        named("command"),
    ),
    "route held in a class attribute, called through self": (
        {
            "helper": helper(
                "class Runner:\n    route = sealed_route.units_route\n"
                "    def open(self, r):\n        return self.route(r)\n"
            ),
            "command": uses("helper.Runner().open(r)"),
        },
        named("command"),
    ),
    "dataclass field default factory": (
        {
            "helper": helper(
                "import dataclasses\n@dataclasses.dataclass\nclass Job:\n"
                "    route: object = dataclasses.field(default_factory=sealed_route.units_route)\n"
            ),
            "command": uses("helper.Job()"),
        },
        named("command"),
    ),
    "conditional expression": (
        {
            "command": IMPORT + "def main(r, fast):\n"
            "    route = sealed_route.units_route if fast else sealed_route.matrix_route\n"
            "    return route(r)\n"
        },
        named("command"),
    ),
    # code run at import, and the record written away from the entry (rule 5b)
    "call at import time, record elsewhere": (
        {"command": IMPORT + "ROUTE = sealed_route.units_route(1)\n" + RECORD},
        entry("command", "ROUTE"),
    ),
    "call at import time, no record": (
        {"command": IMPORT + "sealed_route.units_route(1)\n"},
        named("command"),
    ),
    "loop at import over a route table": (
        {
            "helper": helper("ROUTES = {'units': sealed_route.units_route}\n"),
            "command": "from fleetreserve import helper\nfor route in helper.ROUTES.values():\n"
            "    route(1)\n" + RECORD,
        },
        # `.values()` called on the table at import counts as calling it (over-caution)
        entry("command", "<module>") + entry("command", "route"),
    ),
    "record only in an unused function": (
        {"command": IMPORT + "def main(r):\n    sealed_route.units_route(r)\n" + RECORD},
        entry("command", "main"),
    ),
    "record only in an unused function, main run by the module": (
        {
            "command": IMPORT
            + "def main(r):\n    sealed_route.units_route(r)\n"
            + RECORD
            + "if __name__ == '__main__':\n    main(1)\n"
        },
        entry("command", "<module>"),
    ),
    "a second command in a module whose first writes the record": (
        {
            "command": IMPORT + "def main(r):\n    document = {}\n"
            "    document[ROUTE_KEY] = sealed_route.units_route(r)\n"
            "def other(r):\n    return sealed_route.units_route(r)\n"
        },
        entry("command", "other"),
    ),
    "two functions that call only each other": (
        {
            "command": IMPORT + "def a(r):\n    sealed_route.units_route(r)\n    return b(r)\n"
            "def b(r):\n    return a(r)\n" + RECORD
        },
        entry("command", "a"),
    ),
}


@pytest.mark.parametrize("modules, expected", list(FORMS.values()), ids=list(FORMS))
def test_a_form_of_reaching_the_gate_is_found(modules, expected):
    assert found(**modules) == expected


RE_EXPORTS = {
    # the second review's finding 2: a package `__init__.py` is the package
    "package re-export, the package listed": (
        {
            "zsub.__init__": "from .impl import go\n",
            "zsub.impl": helper("def go(r):\n    return sealed_route.units_route(r)\n"),
            "command": "from fleetreserve.zsub import go\ndef main(r):\n    return go(r)\n",
        },
        {"zsub": "re-exports"},
    ),
    "the package's own `__init__.py`, listed": (
        {
            "__init__": "from .sealed_route import units_route as units\n",
            "command": "from fleetreserve import units\ndef main(r):\n    return units(r)\n",
        },
        {"__init__": "re-exports"},
    ),
    "re-export through two packages": (
        {
            "pa.__init__": "from .impl import go\n",
            "pa.impl": helper("def go(r):\n    return sealed_route.units_route(r)\n"),
            "pb.__init__": "from fleetreserve.pa import go\n",
            "command": "from fleetreserve.pb import go\ndef main(r):\n    return go(r)\n",
        },
        {"pa": "re-exports", "pb": "re-exports"},
    ),
}


@pytest.mark.parametrize("modules, listed", list(RE_EXPORTS.values()), ids=list(RE_EXPORTS))
def test_a_caller_through_a_listed_package_re_export_is_found(modules, listed):
    assert unrecognised(_with(**modules), listed) == named("command")


def test_a_package_that_re_exports_the_gate_is_asked_for_the_record_or_a_reason():
    modules = {"zsub.__init__": "from fleetreserve.sealed_route import units_route\n"}
    assert found(**modules) == named("zsub")


RECOGNISED = {
    "mark through an alias import": "from fleetreserve.sealed_route import mark as m, units_route\n"
    "def main(r):\n    m({}, units_route(r))\n",
    "the key in a dict display": IMPORT + "def main(r):\n"
    "    return {ROUTE_KEY: sealed_route.units_route(r)}\n",
    "the key's text as a keyword": IMPORT + "def main(r):\n"
    "    return dict(sealed_route=sealed_route.units_route(r))\n",
    "the record written by a function of the module that main calls": IMPORT
    + RECORD
    + "def main(r):\n    document = {}\n    sealed_route.units_route(r)\n    record(document)\n"
    "if __name__ == '__main__':\n    main(1)\n",
}


@pytest.mark.parametrize("source", list(RECOGNISED.values()), ids=list(RECOGNISED))
def test_a_module_that_writes_the_record_below_its_entry_is_recognised(source):
    assert found(command=source) == []


NOT_THE_GATE = {
    # the second review's finding 5: a local variable spelled as a function that reaches the gate
    "local variable spelled as a gate function": {
        "helper": helper("def route(r):\n    return sealed_route.units_route(r)\n")
        + "def util(x):\n    route = x\n    return route\n",
        "command": "from fleetreserve import helper\ndef main(r):\n    document = {}\n"
        "    document[ROUTE_KEY] = helper.route(r)\n",
        "other": "from fleetreserve.helper import util\ndef main():\n    return util(1)\n",
    },
    "names of a gate module that do not reach it": {
        "other": IMPORT + "def main():\n    return sealed_route.CHECKS, sealed_route.mark\n",
    },
    "models imported by a name built from a fixed text": {
        "loader": "import importlib\ndef load(name):\n"
        "    module_name = f'fleetreserve.models.{name}'\n"
        "    return importlib.import_module(module_name)\n",
        "models.m": "def make_model():\n    return 1\n",
        "other": "from fleetreserve import loader\ndef main():\n    return loader.load('m')\n",
    },
    "a module that imports a gate module and uses nothing of it": {"other": IMPORT},
    "an instance made at import of a class whose method passes the gate": {
        "helper": helper(RUNNER + "RUNNER = Runner()\n"),
    },
}


@pytest.mark.parametrize("modules", list(NOT_THE_GATE.values()), ids=list(NOT_THE_GATE))
def test_a_module_that_does_not_reach_the_gate_is_not_flagged(modules):
    assert found(**modules) == []


def test_the_over_caution_is_as_stated():
    """A class is one unit: a module naming only a class's constant reaches the gate."""
    command = "from fleetreserve.helper import Runner\ndef main():\n    return Runner.LABEL\n"
    assert found(helper=helper(RUNNER), command=command) == named("command")
