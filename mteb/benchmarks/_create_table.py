from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from mteb.get_tasks import _TASKS_REGISTRY
from mteb.models.model_implementations import MODEL_REGISTRY
from mteb.models.model_meta import _serialize_experiment_kwargs_to_name

if TYPE_CHECKING:
    from collections.abc import Mapping

# Synthetic per-row variant identifier:
# - empty string for base (non-experiment) rows
# - serialized experiment_kwargs (matches on-disk experiment folder name) for variants
# Used as a secondary group / pivot key so each experiment variant of a model becomes
# its own row in the summary/per-task tables. Stored as a separate column (not folded
# into model_name) so the MODEL_REGISTRY lookup in `_attach_model_metadata` stays
# keyed by the canonical model name.
_VARIANT_ID_COL = "_variant_id"


def _serialize_experiment_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # Polars unions all variant keys into one Struct schema and pads absent
    # keys with null. Drop the nulls so the serialized id reflects only the
    # kwargs that actually drove this run — and matches the cleaned dict the
    # API aggregator surfaces.
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if v is not None}
        if not value:
            return ""
    serialized = _serialize_experiment_kwargs_to_name(value)
    return serialized or ""


def _ensure_variant_id(pl_df: pl.DataFrame) -> pl.DataFrame:
    """Add ``_variant_id`` (Utf8) derived from the optional ``experiments`` column.

    No-op when ``experiments`` is absent (older parquet schemas) — the synthetic
    column is still added with an empty string so downstream group_by keys behave
    uniformly. Map_elements is per-row but the variant set is small in practice
    (a handful of ablation kwargs per model), so this stays cheap.
    """
    # Cast model_name out of categorical (pandas writes it that way for memory
    # reasons) so downstream str/list ops in the builders don't trip on the
    # categorical dtype. Cheap when already Utf8. Apply BEFORE the
    # `_VARIANT_ID_COL`-already-present short-circuit so an override frame that
    # carries a pre-built _variant_id still gets the cast.
    if "model_name" in pl_df.columns and pl_df.schema["model_name"] != pl.Utf8:
        pl_df = pl_df.with_columns(pl.col("model_name").cast(pl.Utf8))
    if _VARIANT_ID_COL in pl_df.columns:
        return pl_df
    if "experiments" not in pl_df.columns:
        return pl_df.with_columns(pl.lit("").alias(_VARIANT_ID_COL))
    return pl_df.with_columns(
        # `fill_null("")` because polars' map_elements skips null inputs, but
        # downstream code groups + compares on this column and treats "" as the
        # base (non-experiment) sentinel — see _attach_model_metadata.
        pl.col("experiments")
        .map_elements(_serialize_experiment_value, return_dtype=pl.Utf8)
        .fill_null("")
        .alias(_VARIANT_ID_COL)
    )


@functools.lru_cache(maxsize=4096)
def _training_datasets_cached(model_name: str) -> frozenset[str] | None:
    """Memoized training datasets (with similar tasks) for a model.

    The similar-task graph traversal in ``ModelMeta.get_training_datasets()`` is
    expensive and depends only on the model, so cache it per model name here at the
    leaderboard layer (rather than polluting ``ModelMeta``). Both the summary's
    zero-shot column and ``_filter_models``' zero-shot check share this cache.

    Reads ``MODEL_REGISTRY`` directly (skips the rename check + KeyError path in
    ``get_model_meta``) — this is a hot-path lookup.
    """
    meta = MODEL_REGISTRY.get(model_name)
    if meta is None:
        return None
    training_datasets = meta.get_training_datasets()
    if training_datasets is None:
        return None
    return frozenset(training_datasets)


@functools.lru_cache(maxsize=4096)
def _zero_shot_pct_cached(model_name: str, task_names: tuple[str, ...]) -> int | None:
    """Memoized zero-shot percentage for a model over the given task names."""
    if not task_names:
        return None
    training_datasets = _training_datasets_cached(model_name)
    if training_datasets is None:
        return None
    overlap = training_datasets & set(task_names)
    return int(100 - 100 * (len(overlap) / len(task_names)))


def _is_zero_shot_cached(
    model_name: str, task_name_set: set[str] | frozenset[str]
) -> bool | None:
    """Cached equivalent of ``ModelMeta.is_zero_shot_on(task_names)`` for the leaderboard.

    Returns True if the model was not trained on any of the given tasks, False if it
    was, or None when the model has no training-data info. Reuses
    :func:`_training_datasets_cached`, so repeat calls (e.g. across model-filter
    interactions) avoid recomputing the similar-task graph traversal.
    """
    if not task_name_set:
        return True
    training_datasets = _training_datasets_cached(model_name)
    if training_datasets is None:
        return None
    return not bool(training_datasets & task_name_set)


def _no_results_frame() -> pl.DataFrame:
    """The placeholder frame returned when an empty selection would have no rows."""
    return pl.DataFrame({"No results": ["You can try relaxing your criteria"]})


def _skipna_false_mean(cols: list[str]) -> pl.Expr:
    """Row-wise mean that returns null if any of ``cols`` is null.

    Matches ``pd.DataFrame.mean(axis=1, skipna=False)`` semantics.
    """
    any_null = pl.any_horizontal([pl.col(c).is_null() for c in cols])
    return pl.when(any_null).then(None).otherwise(pl.mean_horizontal(cols))


def _get_borda_rank(score_cols: list[str]) -> pl.Expr:
    """Borda rank for each row across ``score_cols``, as a polars expression.

    Per-column rank (higher score → lower rank number) is converted to a borda count
    (``n - rank``), summed row-wise, and ranked again with ``method="min"``. The row
    count ``n`` is taken from the evaluation context via ``pl.len()`` so the expression
    can be plugged into any ``with_columns`` / ``select`` without an intermediate
    materialisation.
    """
    n = pl.len()
    return (
        pl.sum_horizontal(
            [n - pl.col(c).rank(method="average", descending=True) for c in score_cols]
        )
        .rank(method="min", descending=True)
        .cast(pl.Int64)
    )


def _split_on_capital(s: str) -> str:
    """Splits on capital letters and joins with spaces

    Returns:
        The input string split on capital letters and joined with spaces as a string.
    """
    return " ".join(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s))


def _format_n_parameters(n_parameters: float | int | None) -> float | None:
    """Convert a parameter count to billions with 1M-precision (7M -> 0.007, 1.5B -> 1.5, None -> None)."""
    if n_parameters is None:
        return None
    return round(float(n_parameters) / 1e9, 3)


def _format_max_tokens(max_tokens: float | None) -> float | None:
    if max_tokens is None or max_tokens == np.inf:
        return None
    return float(max_tokens)


def _get_embedding_size(embed_dim: int | Sequence[int] | None) -> int | None:
    if embed_dim is None:
        return None
    if isinstance(embed_dim, int):
        return int(embed_dim)
    if isinstance(embed_dim, Sequence) and len(embed_dim) > 0:
        return int(max(embed_dim))
    return None


def _get_means_per_types(
    task_cols: list[str],
) -> tuple[list[pl.Expr], list[str]]:
    """Per-task-type mean expressions for a given task-column set.

    Returns ``(type_exprs, type_cols)``: a list of polars expressions (one per task
    type, each already aliased via ``_split_on_capital``) and the matching column-name
    list. The expressions can be splatted into a ``select`` / ``with_columns`` on the
    wide task frame so we don't materialise an intermediate ``mean_per_type`` frame
    and don't need a join to bring the type means back into the summary pipeline.
    Means use ``skipna=False`` semantics (matches the prior pandas implementation).
    """
    task_names_per_type: dict[str, list[str]] = defaultdict(list)
    for task_name in task_cols:
        # Read from the registered class to skip instantiation (get_task() runs filter_languages()).
        task_type = _TASKS_REGISTRY[task_name].metadata.type
        task_names_per_type[task_type].append(task_name)

    type_cols: list[str] = []
    type_exprs: list[pl.Expr] = []
    for task_type, tasks in task_names_per_type.items():
        col_name = _split_on_capital(task_type)
        type_cols.append(col_name)
        type_exprs.append(_skipna_false_mean(tasks).alias(col_name))
    return type_exprs, type_cols


_META_STRUCT_FIELDS = {
    "Max Tokens": pl.Float64,
    "Embedding Dimensions": pl.Int64,
    "Total Parameters (B)": pl.Float64,
    "Active Parameters (B)": pl.Float64,
    "Release Date": pl.Utf8,
    "_model_link": pl.Utf8,
}
_META_STRUCT_DTYPE = pl.Struct(_META_STRUCT_FIELDS)
_META_STRUCT_DTYPE_WITH_ZS = pl.Struct({**_META_STRUCT_FIELDS, "Zero-shot": pl.Int64})


def _meta_dict_from_modelmeta_struct(mm: Mapping[str, Any]) -> dict[str, Any]:
    """Project a parquet `model_meta` struct row into the summary meta shape.

    Mirrors :func:`_static_model_meta`'s per-MODEL_REGISTRY entry but reads the
    fields out of the per-experiment ``ModelMeta.to_dict()`` we stored in the
    parquet. Used as a per-variant override so e.g. a variant with a different
    `embed_dim` shows the variant's number rather than the base model's.
    """
    active = mm.get("n_active_parameters_override") or mm.get("n_parameters")
    return {
        "Max Tokens": _format_max_tokens(mm.get("max_tokens")),
        "Embedding Dimensions": _get_embedding_size(mm.get("embed_dim")),
        "Total Parameters (B)": _format_n_parameters(mm.get("n_parameters")),
        "Active Parameters (B)": _format_n_parameters(active),
        "Release Date": str(mm.get("release_date")) if mm.get("release_date") else None,
        "_model_link": mm.get("reference"),
    }


def _build_variant_overrides(
    pl_df: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Per-``(model_name, variant_id)`` summary-meta overrides from the parquet.

    Sourced from the ``model_meta`` struct column (populated by
    :meth:`BenchmarkResults._build_pre_agg_df` for experiment rows). Variant
    rows whose ``model_meta`` snapshot differs from MODEL_REGISTRY's base
    take their numeric meta from the variant; missing keys fall back to the
    base via ``_attach_model_metadata``.
    """
    if "model_meta" not in pl_df.columns or _VARIANT_ID_COL not in pl_df.columns:
        return {}
    rows = (
        pl_df.lazy()
        .filter(pl.col("model_meta").is_not_null())
        .select("model_name", _VARIANT_ID_COL, "model_meta")
        .unique(subset=["model_name", _VARIANT_ID_COL])
        .collect()
    )
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows.iter_rows(named=True):
        mm = r["model_meta"]
        if not mm:
            continue
        overrides[(r["model_name"], r[_VARIANT_ID_COL])] = (
            _meta_dict_from_modelmeta_struct(mm)
        )
    return overrides


@functools.lru_cache(maxsize=1)
def _static_model_meta() -> dict[str, dict[str, Any]]:
    """Cached per-model metadata dict keyed by ``model_name``.

    Built once from ``MODEL_REGISTRY`` (which is static after import) so that
    repeat leaderboard renders reuse the same dict objects instead of
    re-constructing one per model on every call to :func:`_attach_model_metadata`.
    Zero-shot is not stored here — it depends on the active task set and is
    layered on per call.
    """
    return {
        name: {
            "Max Tokens": _format_max_tokens(m.max_tokens),
            "Embedding Dimensions": _get_embedding_size(m.embed_dim),
            "Total Parameters (B)": _format_n_parameters(m.n_parameters),
            "Active Parameters (B)": _format_n_parameters(m.n_active_parameters),
            "Release Date": str(m.release_date) if m.release_date else None,
            "_model_link": m.reference,
        }
        for name, m in MODEL_REGISTRY.items()
    }


def _attach_model_metadata(
    joint_table: pl.DataFrame,
    task_names_key: tuple[str, ...] | None = None,
    variant_overrides: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> pl.DataFrame:
    """Filter to models with valid metadata and attach the standard summary columns.

    Inner-joins meta columns (``Max Tokens``, ``Embedding Dimensions``, ``Total/Active
    Parameters (B)``, ``Release Date``) onto ``joint_table`` (which must have a
    ``model_name`` column), replaces ``model_name`` with a markdown-linked ``Model``
    column, and optionally adds a ``Zero-shot`` column when ``task_names_key`` is
    provided (None → -1 to mirror the previous ``.fillna(-1)``).

    When ``joint_table`` carries a non-empty ``_variant_id`` column (experiment
    variant rows), the short name in the ``Model`` markdown gets a ``" (id)"``
    suffix so the display disambiguates multiple rows for the same base model.
    The variant id column is preserved on the output so downstream code can
    surface the variant kwargs separately.

    ``variant_overrides`` maps ``(model_name, variant_id)`` to a meta dict that
    overlays the MODEL_REGISTRY base — used to surface per-variant numeric
    attributes (different ``embed_dim``, ``max_tokens``, etc.) when the
    experiment's ``model_meta.json`` differs from the base model.
    """
    meta_lookup = _static_model_meta()
    meta_dtype = (
        _META_STRUCT_DTYPE_WITH_ZS if task_names_key is not None else _META_STRUCT_DTYPE
    )
    has_variants = _VARIANT_ID_COL in joint_table.columns
    overrides = variant_overrides or {}
    # Polars pivot can revert model_name back to Categorical from upstream
    # frames; the downstream str/list ops here need Utf8.
    if joint_table.schema.get("model_name") != pl.Utf8:
        joint_table = joint_table.with_columns(pl.col("model_name").cast(pl.Utf8))

    def _resolve(name: str, variant_id: str) -> dict[str, Any] | None:
        base = meta_lookup.get(name)
        override = overrides.get((name, variant_id)) if variant_id else None
        if base is None and override is None:
            return None
        merged: dict[str, Any] = {}
        if base is not None:
            merged.update(base)
        if override is not None:
            # Overlay only non-None override values so a missing variant field
            # falls back to the base. _model_link override of None falls through.
            for k, v in override.items():
                if v is not None:
                    merged[k] = v
        if task_names_key is not None:
            z = _zero_shot_pct_cached(name, task_names_key)
            merged["Zero-shot"] = -1 if z is None else z
        return merged

    if has_variants:

        def _fetch(struct_series: pl.Series) -> pl.Series:
            return pl.Series(
                "_meta",
                [
                    _resolve(s["model_name"], s[_VARIANT_ID_COL] or "")
                    for s in struct_series
                ],
                dtype=meta_dtype,
            )

        out = (
            joint_table.with_columns(
                _key=pl.struct(["model_name", _VARIANT_ID_COL]),
            )
            .with_columns(
                _meta=pl.col("_key").map_batches(_fetch, return_dtype=meta_dtype),
            )
            .drop("_key")
        )
    else:

        def _fetch_name_only(names: pl.Series) -> pl.Series:
            return pl.Series(
                "_meta", [_resolve(n, "") for n in names], dtype=meta_dtype
            )

        out = joint_table.with_columns(
            _meta=pl.col("model_name").map_batches(
                _fetch_name_only, return_dtype=meta_dtype
            ),
        )

    out = (
        out.filter(pl.col("_meta").is_not_null())
        .unnest("_meta")
        .with_columns(
            pl.col("model_name").str.split("/").list.last().alias("_short_name"),
        )
    )
    if has_variants:
        # When the row is a variant, append the serialized kwargs id so the same
        # base model shows as multiple distinguishable rows in the leaderboard.
        out = out.with_columns(
            pl.when(pl.col(_VARIANT_ID_COL) != "")  # noqa: PLC1901
            .then(pl.col("_short_name") + " (" + pl.col(_VARIANT_ID_COL) + ")")
            .otherwise(pl.col("_short_name"))
            .alias("_short_name")
        )
    out = out.with_columns(
        pl.when(pl.col("_model_link").is_not_null())
        .then("[" + pl.col("_short_name") + "](" + pl.col("_model_link") + ")")
        .otherwise(pl.col("_short_name"))
        .alias("Model"),
    ).drop(["_model_link", "_short_name", "model_name"])
    return out


def _create_summary_table_from_benchmark_results(
    pl_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create summary table from a long polars pre-aggregation frame.

    Stays in polars throughout (aggregation, pivot, type-means, borda, model metadata,
    markdown link, sort, rename); converts to pandas only at the return boundary so the
    leaderboard's pandas Styler can consume it.

    Returns a DataFrame with one row per model containing summary statistics
    and task type averages.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    pl_df = _ensure_variant_id(pl_df)
    variant_overrides = _build_variant_overrides(pl_df)
    per_task = (
        pl_df.group_by(["model_name", _VARIANT_ID_COL, "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index=["model_name", _VARIANT_ID_COL], values="score")
    )
    task_cols = [
        c for c in per_task.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    type_exprs, type_cols = _get_means_per_types(task_cols)

    joint_table = (
        per_task.select(
            "model_name",
            _VARIANT_ID_COL,
            *type_exprs,
            _skipna_false_mean(task_cols).alias("Mean (Task)"),
            _get_borda_rank(task_cols).alias("Rank (Borda)"),
        )
        .with_columns(_skipna_false_mean(type_cols).alias("Mean (TaskType)"))
        .sort("Rank (Borda)")
    )

    joint_table = _attach_model_metadata(
        joint_table,
        task_names_key=tuple(sorted(task_cols)),
        variant_overrides=variant_overrides,
    )

    final_cols = [
        "Rank (Borda)",
        "Model",
        _VARIANT_ID_COL,
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Task)",
        "Mean (TaskType)",
        *type_cols,
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])


def _create_per_task_table_from_benchmark_results(
    pl_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create per-task table from a long polars pre-aggregation frame.

    All aggregation, ranking, and sorting runs in polars; the result is converted to
    pandas only at the return boundary (the leaderboard's Styler is pandas-based).

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.

    Returns:
        DataFrame with per-task scores, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    pl_df = _ensure_variant_id(pl_df)
    per_task = (
        pl_df.group_by(["model_name", _VARIANT_ID_COL, "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index=["model_name", _VARIANT_ID_COL], values="score")
    )
    task_cols = [
        c for c in per_task.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]
    if not task_cols:
        return _no_results_frame()

    # Drop models whose task scores are all null.
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    per_task = (
        per_task.sort(_get_borda_rank(task_cols))
        .with_columns(pl.col("model_name").str.split("/").list.last().alias("_short"))
        .with_columns(
            pl.when(pl.col(_VARIANT_ID_COL) != "")  # noqa: PLC1901
            .then(pl.col("_short") + " (" + pl.col(_VARIANT_ID_COL) + ")")
            .otherwise(pl.col("_short"))
            .alias("Model")
        )
        .drop(["model_name", "_short"])
        # Keep ``_variant_id`` on the output so downstream consumers (the API
        # aggregator) can key per-task scores by (model, variant) without
        # reparsing the disambiguated Model markdown.
        .select(["Model", _VARIANT_ID_COL, *task_cols])
    )
    return per_task


def _create_per_language_table_from_benchmark_results(
    pl_df: pl.DataFrame,
    language_view: list[str] | Literal["all"],
) -> pl.DataFrame:
    """Create per-language table from a long polars pre-aggregation frame.

    Returns a DataFrame with one row per model and one column per language.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``language`` (list[str]),
            and ``score`` columns.
        language_view: List of languages to include, or ``"all"`` for every language
            present in the results.

    Returns:
        DataFrame with per-language scores, ready for styling in the leaderboard.
    """
    if language_view != "all" and not isinstance(language_view, list):
        raise ValueError("language_view must be a list of languages or 'all'")

    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    pl_df = _ensure_variant_id(pl_df)
    # Lazy pipeline so polars can fuse explode + filter + group_by. Project only
    # the columns we need so the explode has narrower rows. When a language subset
    # is selected, push the predicate *before* the explode by keeping only rows
    # whose language list intersects the selection — this avoids materialising
    # exploded rows we'll discard.
    lazy = pl_df.lazy().select("model_name", _VARIANT_ID_COL, "language", "score")
    if language_view != "all":
        lazy = lazy.filter(
            pl.col("language").list.eval(pl.element().is_in(language_view)).list.any()
        )
    lazy = lazy.explode("language").drop_nulls("language")
    if language_view != "all":
        lazy = lazy.filter(pl.col("language").is_in(language_view))
    # Streaming engine handles the explode → group_by chain on tens of millions of
    # post-explode rows ~3-4× faster than the default in-memory engine here.
    lang_df = (
        lazy.group_by(["model_name", _VARIANT_ID_COL, "language"])
        .agg(pl.col("score").mean())
        .collect(engine="streaming")
    )
    if lang_df.is_empty():
        return _no_results_frame()

    per_language = lang_df.pivot(
        on="language", index=["model_name", _VARIANT_ID_COL], values="score"
    )
    lang_cols = [
        c for c in per_language.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]
    if not lang_cols:
        return _no_results_frame()
    per_language = per_language.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in lang_cols])
    )
    if per_language.is_empty():
        return _no_results_frame()

    if len(lang_cols) == 1:
        per_language = per_language.sort(lang_cols[0], descending=True, nulls_last=True)
    else:
        per_language = per_language.sort(_get_borda_rank(lang_cols))

    return (
        per_language.with_columns(
            pl.col("model_name").str.split("/").list.last().alias("_short")
        )
        .with_columns(
            pl.when(pl.col(_VARIANT_ID_COL) != "")  # noqa: PLC1901
            .then(pl.col("_short") + " (" + pl.col(_VARIANT_ID_COL) + ")")
            .otherwise(pl.col("_short"))
            .alias("Model")
        )
        .drop(["model_name", _VARIANT_ID_COL, "_short"])
        .select(["Model", *lang_cols])
    )


def _create_summary_table_mean_public_private(  # noqa: PLR0914
    pl_df: pl.DataFrame,
    exclude_private_from_borda: bool = False,
) -> pl.DataFrame:
    """Create summary table that separates public and private task means.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``, ``score``,
            and ``is_public`` columns.
        exclude_private_from_borda: If True, calculate Borda rank using only public tasks.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    pl_df = _ensure_variant_id(pl_df)
    variant_overrides = _build_variant_overrides(pl_df)
    per_task_long = pl_df.group_by(["model_name", _VARIANT_ID_COL, "task_name"]).agg(
        pl.col("score").mean(),
        pl.col("is_public").first(),
    )
    public_tasks = (
        per_task_long.filter(pl.col("is_public"))
        .get_column("task_name")
        .unique()
        .to_list()
    )
    private_tasks = (
        per_task_long.filter(~pl.col("is_public"))
        .get_column("task_name")
        .unique()
        .to_list()
    )
    per_task = per_task_long.pivot(
        on="task_name", index=["model_name", _VARIANT_ID_COL], values="score"
    )
    task_cols = [
        c for c in per_task.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    type_exprs, type_cols = _get_means_per_types(task_cols)

    public_present = [c for c in public_tasks if c in task_cols]
    private_present = [c for c in private_tasks if c in task_cols]
    borda_cols = (
        public_present if exclude_private_from_borda and public_present else task_cols
    )

    public_mean_expr = (
        _skipna_false_mean(public_present).alias("Mean (Public)")
        if public_present
        else pl.lit(None).cast(pl.Float64).alias("Mean (Public)")
    )
    private_mean_expr = (
        _skipna_false_mean(private_present).alias("Mean (Private)")
        if private_present
        else pl.lit(None).cast(pl.Float64).alias("Mean (Private)")
    )

    joint_table = per_task.select(
        "model_name",
        _VARIANT_ID_COL,
        *type_exprs,
        public_mean_expr,
        private_mean_expr,
        _get_borda_rank(borda_cols).alias("Rank (Borda)"),
    ).sort("Rank (Borda)")

    joint_table = _attach_model_metadata(
        joint_table,
        task_names_key=tuple(sorted(task_cols)),
        variant_overrides=variant_overrides,
    )

    final_cols = [
        "Rank (Borda)",
        "Model",
        _VARIANT_ID_COL,
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Public)",
        "Mean (Private)",
        *type_cols,
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])


def _create_summary_table_mean_subset(
    pl_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create summary table where each task-language subset is weighted equally.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            ``subset``, and ``score`` columns.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    pl_df = _ensure_variant_id(pl_df)
    variant_overrides = _build_variant_overrides(pl_df)
    # Per-task mean (for per-type aggregation) and per-(task,subset) mean (for borda).
    per_subset_long = pl_df.group_by(
        ["model_name", _VARIANT_ID_COL, "task_name", "subset"]
    ).agg(pl.col("score").mean())
    per_task = (
        per_subset_long.group_by(["model_name", _VARIANT_ID_COL, "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index=["model_name", _VARIANT_ID_COL], values="score")
    )
    task_cols = [
        c for c in per_task.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    type_exprs, type_cols = _get_means_per_types(task_cols)

    # Mean over all subset rows per (model, variant) (each task-language subset weighted equally).
    overall_subset_mean = per_subset_long.group_by(["model_name", _VARIANT_ID_COL]).agg(
        pl.col("score").mean().alias("Mean (Subset)")
    )
    # Borda over per-(task, subset) columns. Pivot creates "task__subset"-shaped names,
    # but the exact names don't matter — we only need the score columns for ranking.
    per_subset_wide = per_subset_long.with_columns(
        (pl.col("task_name") + "::" + pl.col("subset")).alias("_ts")
    ).pivot(on="_ts", index=["model_name", _VARIANT_ID_COL], values="score")
    subset_cols = [
        c for c in per_subset_wide.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]

    joint_table = (
        per_task.select("model_name", _VARIANT_ID_COL, *type_exprs)
        .join(overall_subset_mean, on=["model_name", _VARIANT_ID_COL], how="left")
        .join(
            per_subset_wide.select(
                "model_name",
                _VARIANT_ID_COL,
                _get_borda_rank(subset_cols).alias("Rank (Borda)"),
            ),
            on=["model_name", _VARIANT_ID_COL],
            how="left",
        )
        .sort("Mean (Subset)", descending=True, nulls_last=True)
    )

    joint_table = _attach_model_metadata(
        joint_table,
        task_names_key=tuple(sorted(task_cols)),
        variant_overrides=variant_overrides,
    )

    final_cols = [
        "Rank (Borda)",
        "Model",
        _VARIANT_ID_COL,
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Subset)",
        *type_cols,
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])


def _create_summary_table_mean_task_type(
    pl_df: pl.DataFrame,
    mean_column_name: str = "Mean (TaskType)",
    sort_by: str | None = None,
) -> pl.DataFrame:
    """Create summary table where the overall mean is the mean of per-task-type means.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.
        mean_column_name: Name for the mean-by-task-type column. Defaults to "Mean (TaskType)".
        sort_by: Column to sort the rows by (and to populate ``Rank``). When
            ``None`` falls back to ``mean_column_name`` (historical behaviour).
            Pass a non-None value when the benchmark wants to rank by a
            column that differs from its primary mean column.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    pl_df = _ensure_variant_id(pl_df)
    variant_overrides = _build_variant_overrides(pl_df)
    per_task = (
        pl_df.group_by(["model_name", _VARIANT_ID_COL, "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index=["model_name", _VARIANT_ID_COL], values="score")
    )
    task_cols = [
        c for c in per_task.columns if c not in {"model_name", _VARIANT_ID_COL}
    ]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    type_exprs, type_cols = _get_means_per_types(task_cols)

    sort_col = sort_by or mean_column_name
    joint_table = (
        per_task.select(
            "model_name",
            _VARIANT_ID_COL,
            *type_exprs,
            _get_borda_rank(task_cols).alias("Rank (Borda)"),
        )
        .with_columns(_skipna_false_mean(type_cols).alias(mean_column_name))
        .sort(sort_col, descending=True, nulls_last=True)
        .with_columns((pl.int_range(0, pl.len()) + 1).cast(pl.Int64).alias("Rank"))
    )

    joint_table = _attach_model_metadata(
        joint_table,
        task_names_key=tuple(sorted(task_cols)),
        variant_overrides=variant_overrides,
    )

    # Renames specific to mean-task-type variants (Vidore/MIEB).
    renames: dict[str, str] = {}
    if "Any Any Multilingual Retrieval" in joint_table.columns:
        renames["Any Any Multilingual Retrieval"] = "Multilingual Retrieval"
    if "Any Any Retrieval" in joint_table.columns:
        renames["Any Any Retrieval"] = "Retrieval"
    if renames:
        joint_table = joint_table.rename(renames)
        type_cols = [renames.get(c, c) for c in type_cols]

    final_cols = [
        "Rank",
        "Model",
        _VARIANT_ID_COL,
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        mean_column_name,
        *type_cols,
        "Rank (Borda)",
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])
