from __future__ import annotations

PUBLIC_ALIGN_MODES: tuple[str, ...] = ("se3", "se3-original", "posyaw", "sim3")

SE3_ALIGN_ALIASES: frozenset[str] = frozenset(
    {
        "se3",
        "epa_step3",
        "epa_se3",
        "epa_se3_eval",
    }
)
SE3R_ALIGN_ALIASES: frozenset[str] = frozenset(
    {
        "se3r",
        "epa_se3r",
        "rotation_first_se3",
    }
)
SE3_ORIGINAL_ALIGN_ALIASES: frozenset[str] = frozenset(
    {
        "se3-original",
        "se3_original",
        # Accept the originally requested spelling without making it canonical.
        "se3-orginal",
        "se3_orginal",
        "position_first_se3",
        "umeyama_se3",
        "epa_se3_original",
    }
)
POSYAW_ALIGN_ALIASES: frozenset[str] = frozenset({"posyaw", "epa_posyaw"})
SIM3_ALIGN_ALIASES: frozenset[str] = frozenset(
    {
        "sim3",
        "epa_sim3",
        "ov_sim3",
        "epica_sim3",
        "epica_sim3_stable",
        "epica_sim3_joint",
        "epica_sim3_trimmed",
        "epa_sim3_v1",
        "epa_sim3_v2",
        "epica_anchor_sim3",
    }
)

SPECIAL_EVAL_ALIGN_MODES: frozenset[str] = frozenset({"none", "origin", "scale"})
LEGACY_ORIGIN_ALIGN_MODES: frozenset[str] = frozenset({"se3single", "posyawsingle"})
COMPAT_ALIGN_MODES: tuple[str, ...] = tuple(
    sorted(
        SE3_ALIGN_ALIASES
        | SE3R_ALIGN_ALIASES
        | SE3_ORIGINAL_ALIGN_ALIASES
        | POSYAW_ALIGN_ALIASES
        | SIM3_ALIGN_ALIASES
        | SPECIAL_EVAL_ALIGN_MODES
        | LEGACY_ORIGIN_ALIGN_MODES
    )
)


def _clean_mode(raw_mode: str | None) -> str:
    return str(raw_mode or "").strip().lower()


def public_align_mode(raw_mode: str | None, *, default: str = "se3", strict: bool = False) -> str:
    """Return the public alignment mode: se3, se3-original, posyaw, or sim3.

    Compatibility aliases are accepted here, but internal/debug modes are not
    exposed to callers that consume this public value.
    """

    mode = _clean_mode(raw_mode)
    if not mode:
        return default
    if mode in SPECIAL_EVAL_ALIGN_MODES:
        return default
    if mode in SE3_ALIGN_ALIASES:
        return "se3"
    if mode in SE3R_ALIGN_ALIASES:
        return "se3"
    if mode in SE3_ORIGINAL_ALIGN_ALIASES:
        return "se3-original"
    if mode in POSYAW_ALIGN_ALIASES:
        return "posyaw"
    if mode in SIM3_ALIGN_ALIASES:
        return "sim3"
    if strict:
        raise ValueError(f"Unsupported alignment mode: {raw_mode}")
    return mode


def resolve_metric_eval_align_mode(
    raw_mode: str | None,
    *,
    default: str = "none",
    collapse_sim3_aliases: bool = False,
    legacy_origin: bool = True,
    step3_as_se3: bool = False,
    strict: bool = False,
) -> str:
    """Normalize a CLI/requested alignment mode for metric evaluation.

    This preserves experimental Sim3 solver aliases by default so advanced
    tools can still request them explicitly. Public wrappers can set
    ``collapse_sim3_aliases=True`` to expose only the public ``sim3`` behavior.
    """

    mode = _clean_mode(raw_mode)
    if not mode:
        return default
    if mode == "epa_step3":
        return "se3" if bool(step3_as_se3) else "none"
    if mode in {"epa_se3", "epa_se3_eval"}:
        return "se3"
    if mode in SE3R_ALIGN_ALIASES:
        return "se3"
    if mode in SE3_ORIGINAL_ALIGN_ALIASES:
        return "se3-original"
    if mode == "epa_posyaw":
        return "posyaw"
    if mode == "epa_sim3":
        return "sim3"
    if legacy_origin and mode in LEGACY_ORIGIN_ALIGN_MODES:
        return "origin"
    if collapse_sim3_aliases and mode in SIM3_ALIGN_ALIASES:
        return "sim3"
    if mode in COMPAT_ALIGN_MODES:
        return mode
    if strict:
        raise ValueError(f"Unsupported alignment mode: {raw_mode}")
    return mode
