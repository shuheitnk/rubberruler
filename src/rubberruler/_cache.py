"""The run cache in cache_dir, and how algorithms are told apart.

Each finished task is one Parquet file in ``cache_dir/runs``. Only the parent
process writes, under a temporary name that ``os.replace`` then makes visible, so no
lock is needed. An algorithm is identified by its name, its source, its default and
closure values, and the global values it reads.
"""

import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import site
import sys
import sysconfig
import tokenize
import types
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from rubberruler._errors import error

CACHE_FORMAT = 1
RUNS_FOLDER = "runs"
MAX_DEPTH = 50
SCHEMA = pa.schema(
    [
        ("instance_id", pa.string()),
        ("seed", pa.int64()),
        ("algorithm", pa.string()),
        ("identifier", pa.string()),
        ("budget", pa.int64()),
        ("run_number", pa.int64()),
        ("ioh_version", pa.string()),
        ("cache_format", pa.int64()),
        ("best", pa.float64()),
        ("evaluations", pa.int64()),
        ("failed", pa.bool_()),
        ("message", pa.string()),
    ]
)


@dataclass(frozen=True)
class RunResult:
    """One finished run: the smallest value within the budget and how it ended."""

    instance_id: str
    seed: int
    algorithm: str
    budget: int
    run_number: int
    best: float
    evaluations: int
    failed: bool
    message: str


@dataclass(frozen=True)
class Identity:
    """The identifier of an algorithm, and the warning if it is not fully identified."""

    identifier: str
    warning: str | None


class RunStore:
    """Finished runs in ``directory/runs``, one Parquet file per finished task."""

    def __init__(self, directory: Path) -> None:
        self.folder = directory / RUNS_FOLDER

    def load(
        self, identifiers: Mapping[str, str]
    ) -> dict[tuple[str, int, str, int, int], RunResult]:
        """The cached runs of these algorithms.

        The keys are (instance_id, seed, algorithm, budget, run_number). Runs of
        another identifier, ioh version, or cache format are not used. A cache that
        cannot be read counts as empty, so its runs are computed again.
        """
        if not self.folder.is_dir():
            return {}
        try:
            dataset = ds.dataset(
                self.folder, format="parquet", schema=SCHEMA, exclude_invalid_files=True
            )
            table = dataset.to_table(
                filter=(ds.field("cache_format") == CACHE_FORMAT)
                & (ds.field("ioh_version") == ioh_version())
                & ds.field("identifier").isin(sorted(set(identifiers.values())))
            )
        except (OSError, pa.ArrowException):
            return {}
        columns = {name: table.column(name).to_pylist() for name in SCHEMA.names}
        runs: dict[tuple[str, int, str, int, int], RunResult] = {}
        for row in zip(*(columns[name] for name in SCHEMA.names), strict=True):
            values = dict(zip(SCHEMA.names, row, strict=True))
            if identifiers.get(values["algorithm"]) != values["identifier"]:
                continue
            result = RunResult(
                instance_id=values["instance_id"],
                seed=values["seed"],
                algorithm=values["algorithm"],
                budget=values["budget"],
                run_number=values["run_number"],
                best=values["best"],
                evaluations=values["evaluations"],
                failed=values["failed"],
                message=values["message"],
            )
            key = (
                result.instance_id,
                result.seed,
                result.algorithm,
                result.budget,
                result.run_number,
            )
            runs.setdefault(key, result)
        return runs

    def save(
        self,
        results: Sequence[RunResult],
        identifiers: Mapping[str, str],
        stored_ids: Mapping[str, str] | None = None,
    ) -> None:
        """Write the runs of one task as a new Parquet file.

        ``stored_ids`` maps an instance_id to the id under which its runs are
        stored (for transformed instances and objectives of the user); other ids
        are stored as they are.
        """
        ids = stored_ids or {}
        if not results:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "instance_id": [
                    ids.get(result.instance_id, result.instance_id)
                    for result in results
                ],
                "seed": [result.seed for result in results],
                "algorithm": [result.algorithm for result in results],
                "identifier": [identifiers[result.algorithm] for result in results],
                "budget": [result.budget for result in results],
                "run_number": [result.run_number for result in results],
                "ioh_version": [ioh_version()] * len(results),
                "cache_format": [CACHE_FORMAT] * len(results),
                "best": [result.best for result in results],
                "evaluations": [result.evaluations for result in results],
                "failed": [result.failed for result in results],
                "message": [result.message for result in results],
            },
            schema=SCHEMA,
        )
        name = uuid.uuid4().hex
        temporary = self.folder / f".{name}.parquet"
        pq.write_table(table, temporary)
        os.replace(temporary, self.folder / f"{name}.parquet")


def check_cache_dir(cache_dir: object) -> Path | None:
    """Return cache_dir as a Path (None stays None); stop if it cannot be written."""
    if cache_dir is None:
        return None
    if not isinstance(cache_dir, str | os.PathLike):
        raise error(
            f"cache_dir is a {type(cache_dir).__name__}.",
            why="cache_dir must be the path of a folder, or None.",
            fix='pass a path such as cache_dir=".rubberruler", or cache_dir=None.',
        )
    path = Path(cache_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-check-{uuid.uuid4().hex}"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exception:
        raise error(
            f"cache_dir={str(path)!r} cannot be written "
            f"({type(exception).__name__}: {exception}).",
            why="rubberruler keeps the finished runs there, so that it can resume.",
            fix="pass a folder that you can write to, or cache_dir=None to keep "
            "nothing.",
        ) from exception
    return path


def identify(name: str, algorithm: object) -> Identity:
    """The identifier of an algorithm: blake2b (16 bytes, hex) of its encoding."""
    encoder = Encoder()
    encoding = digest(b"algorithm", name.encode(), encoder.callable(algorithm, 0))
    identifier = hashlib.blake2b(encoding, digest_size=16).hexdigest()
    warning = None
    if encoder.source_missing:
        warning = f"{name}: cached by name only (source not available)"
    elif encoder.value_skipped:
        warning = (
            f"{name}: cached by name and source only (some values could not be hashed)"
        )
    return Identity(identifier, warning)


class Encoder:
    """Turns values into bytes by fixed rules per type (no pickle, no id or hash).

    Each value becomes the blake2b digest of a type tag and the encodings of its
    parts, so the result is the same in every process.
    """

    def __init__(self) -> None:
        self.source_missing = False
        self.value_skipped = False
        self._seen: set[int] = set()
        self._path: set[int] = set()

    def value(self, value: object, depth: int) -> bytes:
        """Encode a value by the rule for its type."""
        if depth > MAX_DEPTH:
            return self.skip(value)
        if isinstance(value, np.generic):
            return digest(b"numpy", str(value.dtype).encode(), repr(value).encode())
        if value is None or isinstance(value, bool | int | float | str | bytes):
            return digest(type(value).__name__.encode(), repr(value).encode())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return self.skip(value)
            return digest(
                b"ndarray",
                str(value.dtype).encode(),
                repr(value.shape).encode(),
                np.ascontiguousarray(value).tobytes(),
            )
        if isinstance(value, types.ModuleType):
            return self.library(value.__name__, "")
        if isinstance(value, list | tuple | dict | set | frozenset):
            return self.container(value, depth)
        if callable(value):
            return self.callable(value, depth)
        return self.skip(value)

    def container(
        self,
        value: list[object]
        | tuple[object, ...]
        | dict[object, object]
        | set[object]
        | frozenset[object],
        depth: int,
    ) -> bytes:
        if id(value) in self._path:
            return self.skip(value)
        self._path.add(id(value))
        try:
            if isinstance(value, dict):
                parts = sorted(
                    self.value(key, depth + 1) + self.value(item, depth + 1)
                    for key, item in value.items()
                )
            elif isinstance(value, set | frozenset):
                parts = sorted(self.value(item, depth + 1) for item in value)
            else:
                parts = [self.value(item, depth + 1) for item in value]
        finally:
            self._path.discard(id(value))
        return digest(type(value).__name__.encode(), *parts)

    def callable(self, function: object, depth: int) -> bytes:
        """A function, partial, bound method, class, or callable object."""
        if depth > MAX_DEPTH:
            return self.skip(function)
        if isinstance(function, functools.partial):
            return digest(
                b"partial",
                self.callable(function.func, depth + 1),
                self.value(function.args, depth + 1),
                self.value(function.keywords, depth + 1),
            )
        if isinstance(function, types.MethodType):
            return digest(
                b"method",
                self.callable(function.__func__, depth + 1),
                self.state(function.__self__, depth + 1),
            )
        module = getattr(function, "__module__", None)
        module = module if isinstance(module, str) else None
        qualname = str(getattr(function, "__qualname__", type(function).__qualname__))
        if isinstance(function, types.FunctionType):
            if module is not None and is_library(module):
                return self.library(module, qualname)
            return self.function(function, depth)
        if isinstance(function, type):
            if module is not None and is_library(module):
                return self.library(module, qualname)
            return self.user_class(function, depth)
        if isinstance(function, types.BuiltinFunctionType | types.BuiltinMethodType):
            self.source_missing = True
            return self.library(module or "builtins", qualname)
        kind = type(function)
        if is_library(kind.__module__):
            return self.library(kind.__module__, kind.__qualname__)
        return digest(
            b"object",
            self.callable(kind.__call__, depth + 1),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            self.state(function, depth + 1),
        )

    def function(self, function: types.FunctionType, depth: int) -> bytes:
        """A user function: source, defaults, closure values, and global values."""
        if id(function) in self._seen:
            return digest(b"seen", function.__qualname__.encode())
        self._seen.add(id(function))
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError, SyntaxError, IndexError, tokenize.TokenError):
            # inspect reads the file of the code; another text there breaks tokenize.
            self.source_missing = True
            source = ""
        code = function.__code__
        closure = [
            digest(b"cell", name.encode(), self.value(cell.cell_contents, depth + 1))
            for name, cell in zip(
                code.co_freevars, function.__closure__ or (), strict=True
            )
            if not _is_empty(cell)
        ]
        names = global_names(code)
        references = [
            digest(
                b"global",
                name.encode(),
                self.global_value(function.__globals__[name], names, depth + 1),
            )
            for name in sorted(names)
            if name in function.__globals__
        ]
        return digest(
            b"function",
            function.__qualname__.encode(),
            source.encode(),
            self.value(function.__defaults__, depth + 1),
            self.value(function.__kwdefaults__, depth + 1),
            digest(b"closure", *closure),
            digest(b"globals", *references),
        )

    def global_value(self, value: object, names: set[str], depth: int) -> bytes:
        """A global value; a user's module counts by the attributes the code names.

        A function that calls ``helpers.step(x)`` names ``helpers`` and ``step``, so
        a change of ``helpers.step`` changes the identifier, as for an imported
        function. A library module counts by its version, as elsewhere.
        """
        if not isinstance(value, types.ModuleType) or is_library(value.__name__):
            return self.value(value, depth)
        attributes = vars(value)
        return digest(
            b"user module",
            value.__name__.encode(),
            *(
                digest(b"attribute", name.encode(), self.value(attributes[name], depth))
                for name in sorted(names)
                if name in attributes
            ),
        )

    def user_class(self, kind: type, depth: int) -> bytes:
        """A user class: its bases and what its namespace holds, methods as functions.

        The class's own source is not used: before Python 3.13, inspect looks for it in
        the file of sys.modules[kind.__module__], which is another file after runpy.
        """
        if id(kind) in self._seen:
            return digest(b"seen", kind.__qualname__.encode())
        self._seen.add(id(kind))
        bases = [self.callable(base, depth + 1) for base in kind.__bases__]
        members = [
            digest(b"member", name.encode(), self.member(value, depth + 1))
            for name, value in sorted(vars(kind).items())
            if name not in {"__dict__", "__weakref__"}
        ]
        return digest(
            b"class",
            kind.__qualname__.encode(),
            digest(b"bases", *bases),
            digest(b"members", *members),
        )

    def member(self, value: object, depth: int) -> bytes:
        """A value in a class namespace; the functions inside descriptors count."""
        if isinstance(value, staticmethod | classmethod):
            return digest(b"wrapped", self.callable(value.__func__, depth + 1))
        if isinstance(value, property):
            parts = (value.fget, value.fset, value.fdel)
            return digest(b"property", *(self.value(part, depth + 1) for part in parts))
        return self.value(value, depth)

    def state(self, instance: object, depth: int) -> bytes:
        """The __dict__ of an object; objects without one cannot be encoded."""
        try:
            attributes = vars(instance)
        except TypeError:
            return self.skip(instance)
        return self.value(dict(attributes), depth)

    def library(self, module: str, qualname: str) -> bytes:
        """A module or a library function or class: its top package and version."""
        version = package_version(module)
        if version is None:
            self.value_skipped = True
            version = "unknown"
        return digest(b"library", version.encode(), qualname.encode())

    def skip(self, value: object) -> bytes:
        self.value_skipped = True
        return digest(b"skipped", type(value).__qualname__.encode())


def digest(tag: bytes, *parts: bytes) -> bytes:
    """blake2b of the tag and the parts, each with its length first."""
    hasher = hashlib.blake2b(digest_size=32)
    hasher.update(len(tag).to_bytes(8, "little") + tag)
    for part in parts:
        hasher.update(len(part).to_bytes(8, "little") + part)
    return hasher.digest()


def global_names(code: types.CodeType) -> set[str]:
    """co_names of the code object and of the code objects nested in it."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= global_names(constant)
    return names


def is_library(module: str | None) -> bool:
    """Whether a module is in the standard library or an installed library.

    A library is in site-packages, or in the folder of a distribution that an
    installer put there, such as ``pip install --target`` or ``uv run --with`` do.
    ``__main__`` is the user's script, even if ``sys.modules["__main__"]`` is later a
    module in site-packages (as after runpy under ``python -m pytest``).
    """
    if module is None or module == "__main__":
        return False
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or top == "builtins":
        return True
    loaded = sys.modules.get(module) or sys.modules.get(top)
    path = getattr(loaded, "__file__", None)
    if not isinstance(path, str):
        return False
    real = os.path.realpath(path)
    if any(real.startswith(folder + os.sep) for folder in library_folders()):
        return True
    return any(
        root is not None and real.startswith(root + os.sep)
        for root in map(installed_folder, _distributions().get(top, []))
    )


@functools.cache
def installed_folder(distribution: str) -> str | None:
    """Where an installer put this distribution, or None if it is the user's code.

    A distribution without a RECORD (such as ``setup.py develop``) or installed as
    editable is the user's own code, whose source must count.
    """
    try:
        found = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    if found.read_text("RECORD") is None:
        return None
    try:
        direct_url = json.loads(found.read_text("direct_url.json") or "{}")
    except ValueError:
        return None
    dir_info = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        return None
    return os.path.realpath(str(found.locate_file("")))


@functools.cache
def library_folders() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    folders = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    folders.update(site.getsitepackages())
    folders.add(site.getusersitepackages())
    return tuple(sorted(os.path.realpath(folder) for folder in folders))


def package_version(module: str) -> str | None:
    """ "name version" of the distribution with this top-level module, if any."""
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or top == "builtins":
        return f"python {platform.python_version()}"
    for distribution in _distributions().get(top, []):
        try:
            return f"{distribution} {importlib.metadata.version(distribution)}"
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@functools.cache
def ioh_version() -> str:
    return importlib.metadata.version("ioh")


@functools.cache
def _distributions() -> Mapping[str, list[str]]:
    return importlib.metadata.packages_distributions()


def _is_empty(cell: types.CellType) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:
        return True
    return False
