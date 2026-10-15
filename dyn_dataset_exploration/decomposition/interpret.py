"""S4 -- score every option against the measurements, then decide who answers.

"Interpreting the measurement is as important as measuring it": an option is not a label
to be looked up but a claim (its ``atoms``) that a measurement either supports, contradicts
or cannot check. A claim that cannot be checked scores nothing rather than being guessed,
and a question is answered deterministically (leaf A) only when the best option is clearly
supported, clearly ahead of the runner-up, and every claim it makes could be checked.
Otherwise the evidence goes to Qwen (leaf B), which sees the frames as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dyn_dataset_exploration.decomposition.measure import TUNING, Result
from dyn_dataset_exploration.decomposition.program import KIND_BY_NAME, QueryProgram, class_key

DECISION: dict[str, float] = {
    "min_score": 0.6,          # the winner must be at least this well supported
    "min_margin": 0.25,        # ... and this far ahead of the runner-up
    "fused_margin": 0.25,      # for a fused kind: the winner's share of the posterior, ahead of the runner-up
    "metric_sigma_floor": 0.30,    # log-space error of the metric estimate itself
    # How far each axis of "moves right and slightly forward" can be trusted (weights of the axis
    # claims). Left/right agrees with the ground truth ~90% of the time in the first camera's axes;
    # forward/backward is at chance in every frame (the benchmark uses "forward" both for heading
    # away and toward the camera), so its weight is calibrated down and options that differ only in
    # depth stay tied and go to the language model, rather than being decided by noise.
    "dir_w_x": 1.0, "dir_w_y": 0.5, "dir_w_z": 0.3,
}

# The soft leaf: Qwen's probability of each option letter (from the frames alone) times the
# measurement's support for it, per kind: log p = a * log p_qwen + b * log(support + eps).
# ``b`` = 0 is Qwen alone; a large ``b`` is the measurement with Qwen breaking ties. Fitted per kind on dev.
FUSION: dict[str, dict[str, float]] = {}


def fuse_scores(support: Mapping[str, float], unverifiable: Sequence[str], logprobs: Mapping[str, float],
                a: float, b: float, eps: float) -> tuple[str, float, dict[str, float]]:
    """The letter with the highest fused log-probability, its margin (posterior share ahead of the
    runner-up), and the posterior. An option whose claim cannot be checked gets no measurement term."""

    total = {}
    for letter, lp in logprobs.items():
        term = 0.0 if letter in unverifiable else b * math.log(support.get(letter, 0.0) + eps)
        total[letter] = a * lp + term
    top = max(total.values())
    z = sum(math.exp(v - top) for v in total.values())
    posterior = {l: math.exp(v - top) / z for l, v in total.items()}
    ranked = sorted(posterior.items(), key=lambda kv: (-kv[1], kv[0]))
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return ranked[0][0], margin, posterior


def fuse_for_kind(kind: str, scores: Sequence[OptionScore], logprobs: Mapping[str, float]) -> tuple[str, float] | None:
    """The soft leaf's answer for one question, or None when the kind has no fitted fusion."""

    params = FUSION.get(kind) or {"a": 1.0, "b": 0.0, "eps": 1.0}     # no fitted fusion: Qwen's probabilities alone
    if not logprobs:
        return None
    letter, margin, _ = fuse_scores({s.letter: s.score for s in scores},
                                    [s.letter for s in scores if s.verdict == "unverifiable"], logprobs,
                                    params["a"], params["b"], params["eps"])
    return letter, margin


# D7 per kind: how sure the measurement has to be before it answers alone. Empty means the global
# DECISION values; calibration fills it where a kind's own numbers do better than the shared ones.
DECISION_BY_KIND: dict[str, dict[str, float]] = {}


def gate(kind: str) -> dict[str, float]:
    return {**DECISION, **DECISION_BY_KIND.get(kind, {})}


# Which frame convention "moves right and forward" is read in, and whether "forward" means
# toward the camera. The generator's convention, not physics: chosen by calibration.
DIRECTION_CONVENTION: dict[str, Any] = {"frame": "camera_rel", "z_toward_camera": False}

# Kinds allowed to answer deterministically. A kind is added once it beats the frames-only
# baseline on the dev split; until then its evidence goes to Qwen.
TRUSTED_KINDS: set[str] = {
    "camera_path", "camera_shake", "cam_subject_range", "subject_direction", "pair_relation",
    "metric_distance", "instance_count", "event_order",
}


# Kinds whose answer distribution is skewed enough that the measurement is best read *with* it:
# the option's support (plus a floor, so a zero score is not a veto) times how often that class of
# claim is right. Chosen per kind by calibration, with priors counted on dev clips only.
FUSE_KINDS: set[str] = set()
FUSE_EPS: dict[str, float] = {}
PRIORS: dict[str, dict[str, int]] = {}


@dataclass
class OptionScore:
    letter: str
    text: str
    score: float
    verdict: str            # match | partial | mismatch | unverifiable
    detail: str


@dataclass
class Decision:
    leaf: str                        # A | B | C
    chosen: str | None
    margin: float
    scores: list[OptionScore]
    reason: str
    measured: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"leaf": self.leaf, "chosen": self.chosen, "margin": round(self.margin, 4),
                "reason": self.reason, "measured": self.measured,
                "scores": [vars(s) for s in self.scores]}


def _verdict(score: float) -> str:
    return "match" if score >= 0.85 else "partial" if score >= 0.3 else "mismatch"


# --------------------------------------------------------------------------- #
# one scorer per answer form: (atoms, measured value) -> (score, detail)
# --------------------------------------------------------------------------- #

def score_camera_motion(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    om, od = atoms["motion"], atoms.get("direction")
    mm, md = v["motion"], v["direction"]
    signed, strength = v.get("signed", {}), v.get("strength", {})
    if om in ("zoom",):
        return None, "zoom is not measured here"
    if om == "complex":
        return (1.0, "measured a mix") if mm == "complex" else (0.1, f"measured {mm}")
    if om == "static":
        return (1.0, "no motion above the floors") if mm == "static" else (0.0, f"measured {mm}")
    if mm == "static":
        return 0.0, "nothing above the floors moved"
    if om == "vertical":
        return (1.0, f"measured {mm}") if mm in ("tilt", "pedestal") else (0.0, f"measured {mm}")
    if om == "pan" and od is None:                       # "Horizontal: panning or scanning"
        if mm == "pan":
            return 1.0, "measured a pan"
        return (0.8, f"measured {mm} (horizontal)") if mm in ("truck", "orbit") else (0.0, f"measured {mm}")
    if om == mm:
        if od is None:
            return 0.85, f"{mm}, direction not stated"
        return (1.0, f"{mm} {md}") if od == md else (0.0, f"measured {mm} {md}, not {od}")
    if mm == "complex":                                  # partly right if this family is active
        active = strength.get(om, 0)
        s = signed.get(om, 0.0)
        toward = {"left": -1, "right": 1, "up": 1, "down": -1, "forward": 1, "backward": -1}.get(od or "", 0)
        if active >= 1 and (toward == 0 or s * toward > 0):
            return 0.35, f"one component of a mix ({om} strength {active:.1f})"
        return 0.0, "measured a mix without this motion"
    if {om, mm} == {"pan", "truck"} and od is not None and od == md:
        return 0.3, f"measured {mm} {md}: rotation and translation are easily confused"
    return 0.0, f"measured {mm}"


def score_shake(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    bands = [(0.0, TUNING["jitter_slight_deg"]), (TUNING["jitter_slight_deg"], TUNING["jitter_shaky_deg"]),
             (TUNING["jitter_shaky_deg"], TUNING["jitter_severe_deg"]), (TUNING["jitter_severe_deg"], math.inf)]
    level = int(atoms["level"])
    lo, hi = bands[level]
    r = max(float(v["rotation_residual_deg"]), 1e-4)
    if lo <= r < hi:
        return 1.0, f"residual {r:.2f} deg is inside the band [{lo:g}, {hi:g})"
    d = math.log(r / hi) if r >= hi else math.log(max(lo, 1e-4) / r)
    return math.exp(-d / 0.5), f"residual {r:.2f} deg is outside the band [{lo:g}, {hi:g}) by a factor {math.exp(d):.2f}"


_TREND_NEIGHBOURS = {
    ("closer", "closer_then_constant"): 0.9, ("farther", "farther_then_constant"): 0.9,
    ("closer_then_farther", "fluctuating"): 0.3, ("farther_then_closer", "fluctuating"): 0.3,
    ("fluctuating", "closer_then_farther"): 0.3, ("fluctuating", "farther_then_closer"): 0.3,
    ("closer", "closer_then_farther"): 0.15, ("farther", "farther_then_closer"): 0.15,
    ("closer_then_farther", "farther"): 0.0, ("constant", "fluctuating"): 0.05,
}


def score_trend(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    if atoms.get("undeterminable") or atoms.get("circling"):
        return None, "a claim a distance alone cannot check"
    if "trend" not in atoms:
        return None, "not a claim about a distance"
    want, got = atoms["trend"], v["trend"]
    net = v.get("net_units") or 0.0
    plateau = v.get("plateau_from")
    if want == got:
        if got in ("closer", "farther") and plateau is not None and plateau <= 0.65 and want == got:
            return 0.9, f"{got}, levelling off from {plateau:.0%} of the window"
        return 1.0, f"measured {got}"
    if want in ("closer_then_constant", "farther_then_constant"):
        base = want.split("_")[0]
        if got == base:
            ok = plateau is not None and plateau <= 0.65
            return (1.0 if ok else 0.4), f"{got}; plateau from {plateau if plateau is None else round(plateau, 2)}"
        return 0.0, f"measured {got}"
    if (got, want) in _TREND_NEIGHBOURS:
        return _TREND_NEIGHBOURS[(got, want)], f"measured {got}, option says {want}"
    if want in ("closer", "farther") and got in ("closer_then_farther", "farther_then_closer"):
        toward = -1 if want == "closer" else 1
        if net * toward > 0:
            return 0.4, f"measured {got} with a net change in the same direction"
    return 0.0, f"measured {got}, not {want}"


def _named_components(comp: np.ndarray, floor: float, dominance: float) -> dict[str, str | None]:
    """Which axes of ``[right, down, forward]`` count as moved, and which way."""

    biggest = float(np.max(np.abs(comp)))
    out: dict[str, str | None] = {"x": None, "y": None, "z": None}
    if biggest < floor:
        return out
    for axis, value, (pos, neg) in zip("xyz", comp, (("right", "left"), ("down", "up"), ("forward", "backward"))):
        if abs(value) >= dominance * biggest:
            out[axis] = pos if value >= 0 else neg
    return out


def direction_words(value: Mapping[str, Any], convention: Mapping[str, Any] | None = None
                    ) -> dict[str, Any]:
    """The measured displacement as direction words in one convention."""

    if value.get("direct"):
        # the camera's own displacement: no reference-frame or "forward" convention to choose
        comp = np.asarray(value["frames"]["levelled0"], dtype=float)
        floor = max(TUNING["cam_dir_floor_scales"] * value["scene_scale"], 3 * value["noise"])
        return {"words": _named_components(comp, floor, TUNING["cam_dir_dominance"]), "components": comp.tolist()}
    conv = dict(DIRECTION_CONVENTION if convention is None else convention)
    comp = np.asarray(value["frames"][conv["frame"]], dtype=float).copy()
    if conv.get("z_toward_camera"):
        comp[2] = -comp[2]
    if conv["frame"] == "image":
        floor = TUNING["image_move_floor"]
        comp_for_floor = comp.copy()
        # x, y are image fractions; z is in scene units -- rank them on separate scales
        named = _named_components(np.array([comp[0], comp[1], 0.0]), floor, TUNING["axis_dominance"])
        zfloor = TUNING["move_floor_scales"] * value["scene_scale"]
        named["z"] = ("forward" if comp[2] > 0 else "backward") if abs(comp[2]) >= zfloor else None
        return {"words": named, "components": comp_for_floor.tolist()}
    floor = max(TUNING["move_floor_scales"] * value["scene_scale"], 3 * value["noise"])
    return {"words": _named_components(comp, floor, TUNING["axis_dominance"]), "components": comp.tolist()}


def score_direction(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    read = direction_words(v)
    words = read["words"]
    if atoms.get("trend"):                               # "It moves closer to the camera"
        z = words["z"]
        toward = "backward" if not DIRECTION_CONVENTION.get("z_toward_camera") else "forward"
        got = ("closer" if z == toward else "farther") if z else "constant"
        return (1.0, f"depth {got}") if got == atoms["trend"] else (0.0, f"depth {got}, not {atoms['trend']}")
    if atoms.get("stationary"):
        moved = [a for a, w in words.items() if w]
        return (1.0, "no axis above its floor") if not moved else (0.0, f"moved along {moved}")
    if atoms.get("random") or atoms.get("lateral") or atoms.get("circling"):
        return None, "a claim this displacement cannot check"
    slight = set(atoms.get("slight", []))
    named = [a for a in "xyz" if a in atoms]
    if not named:
        return None, "no direction words"
    # a subject's forward/backward is close to a guess; the camera's own axes are not
    weight = {a: 1.0 if v.get("direct") else float(DECISION[f"dir_w_{a}"]) for a in "xyz"}
    earned, at_stake, notes = 0.0, 0.0, []
    for axis in named:
        got = words[axis]
        at_stake += weight[axis]
        if got == atoms[axis]:
            earned += weight[axis]
        elif axis in slight and got is None:
            earned += 0.5 * weight[axis]
            notes.append(f"{axis}: 'slightly {atoms[axis]}', measured ~0")
        else:
            notes.append(f"{axis}: option {atoms[axis]}, measured {got}")
    if at_stake <= 0:
        return None, "every named axis is one this measurement cannot check"
    extra = [a for a in "xyz" if a not in atoms and words[a]]
    score = max(0.0, earned / at_stake - 0.15 * sum(weight[a] for a in extra))
    agree = sum(1 for a in named if words[a] == atoms[a])
    detail = f"{agree}/{len(named)} named axes agree" + ("; " + "; ".join(notes) if notes else "") \
        + (f"; unmentioned motion along {extra}" if extra else "")
    return score, detail


def score_metric(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    speed = atoms.get("unit") == "m/s"
    key, unit = ("mps", "m/s") if speed else ("metres", "m")
    if key not in v:
        return None, "no metric scale for this clip"
    measured = float(v[key])
    value = float(atoms["value"])
    if measured <= 0 or value <= 0:
        return None, "non-positive value"
    sigma = math.sqrt(DECISION["metric_sigma_floor"] ** 2 + float(v.get("scale_iqr_over_median") or 0.0) ** 2)
    d = abs(math.log(measured / value))
    return math.exp(-0.5 * (d / sigma) ** 2), f"measured {measured:.2f} {unit} vs {value:g}: log-gap {d:.2f}, sigma {sigma:.2f}"


def score_count(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    diff = abs(int(atoms["count"]) - int(v["count"]))
    return {0: 1.0, 1: 0.3}.get(diff, 0.0), f"counted {v['count']}, option says {atoms['count']}"


def score_order(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
    first = v["first"]
    if first.startswith("one_only") or first == "none":
        return None, "an event was not found, so the order is unknown"
    want = atoms["first"]
    if atoms.get("only") or want in ("one_only", "none"):
        return 0.0, f"both events were found; measured order {first}"
    if want == first:
        return 1.0, f"{first} starts first (onsets {v.get('onsets')})"
    return 0.0, f"measured {first} first, not {want}"


# Categorical answers: option key, measured key, and how much a neighbouring class counts. The
# partial credit only sets how close the runner-up is (and so whether the margin test passes);
# it never makes a wrong class the winner.
_CATEGORICAL: dict[str, tuple[str, str, dict[tuple[str, str], float], set[str]]] = {
    # form: (atoms key, measured key, {(measured, option): credit}, option values that cannot be checked)
    "rhythm": ("rhythm", "rhythm", {("steady", "irregular"): 0.3, ("irregular", "steady"): 0.3,
                                    ("still_then_burst", "speeds_up"): 0.5, ("speeds_up", "still_then_burst"): 0.5,
                                    ("fixed", "shaky"): 0.4, ("shaky", "fixed"): 0.5}, set()),
    "zoom": ("zoom", "zoom", {("in", "mixed"): 0.3, ("out", "mixed"): 0.3, ("mixed", "in"): 0.4, ("mixed", "out"): 0.4}, set()),
    "revisit": ("revisit", "revisit", {("same", "slight"): 0.3, ("slight", "same"): 0.3, ("slight", "different"): 0.3,
                                       ("different", "slight"): 0.3}, {"obscured"}),
    "elevation": ("angle", "angle", {("eye", "high"): 0.2, ("eye", "low"): 0.2, ("high", "eye"): 0.2, ("low", "eye"): 0.2,
                                     ("changing", "high"): 0.15, ("changing", "low"): 0.15, ("changing", "eye"): 0.15}, set()),
    "visibility": ("visible", "visible", {("partial", "full"): 0.3, ("full", "partial"): 0.3, ("out", "partial"): 0.3,
                                          ("partial", "out"): 0.3}, {"unclear"}),
    "size": ("size", "size", {("unchanged", "larger"): 0.15, ("unchanged", "smaller"): 0.15,
                              ("larger", "unchanged"): 0.15, ("smaller", "unchanged"): 0.15,
                              ("fluctuating", "larger"): 0.2, ("fluctuating", "smaller"): 0.2,
                              ("larger", "fluctuating"): 0.2, ("smaller", "fluctuating"): 0.2}, set()),
    "motion_profile": ("motion", "motion", {("continuous", "intermittent"): 0.25, ("intermittent", "continuous"): 0.25,
                                            ("once", "intermittent"): 0.4, ("intermittent", "once"): 0.4,
                                            ("none", "once"): 0.3, ("once", "none"): 0.3}, set()),
}


def _categorical_scorer(form: str):
    atoms_key, value_key, credit, uncheckable = _CATEGORICAL[form]

    def score(atoms: Mapping[str, Any], v: Mapping[str, Any]) -> tuple[float | None, str]:
        want, got = atoms.get(atoms_key), v[value_key]
        if want is None:
            return None, "not a claim about this quantity"
        if want in uncheckable:
            return None, f"'{want}' is a claim this measurement cannot check"
        if want == got:
            return 1.0, f"measured {got}"
        c = credit.get((got, want), 0.0)
        return c, f"measured {got}, option says {want}"

    return score


SCORERS = {
    "camera_motion": score_camera_motion, "shake_level": score_shake, "trend": score_trend,
    "direction": score_direction, "metric": score_metric, "count": score_count, "order": score_order,
    **{form: _categorical_scorer(form) for form in _CATEGORICAL},
}


# --------------------------------------------------------------------------- #
# scoring and the decision
# --------------------------------------------------------------------------- #

def main_result(program: QueryProgram, results: Mapping[str, Result]) -> Result | None:
    """The measurement an option is judged against: the last one the program declares."""

    for m in reversed(program.measurements):
        if m.id in results:
            return results[m.id]
    return None


def score_options(program: QueryProgram, results: Mapping[str, Result]) -> list[OptionScore]:
    result = main_result(program, results)
    scorer = SCORERS.get(program.answer_form)
    out: list[OptionScore] = []
    for spec in program.options:
        if scorer is None or result is None or not result.available or spec.atoms is None:
            why = ("no scorer for this answer form" if scorer is None else
                   "the measurement is unavailable" if result is None or not result.available else
                   "the option makes no claim the parser could read")
            out.append(OptionScore(spec.letter, spec.text, 0.0, "unverifiable", why))
            continue
        score, detail = scorer(spec.atoms, result.value)
        if score is None:
            out.append(OptionScore(spec.letter, spec.text, 0.0, "unverifiable", detail))
        else:
            out.append(OptionScore(spec.letter, spec.text, float(score), _verdict(score), detail))
    return out


def _fused_decision(program: QueryProgram, result: Result, scores: list[OptionScore], kind: str,
                    measured: dict[str, Any]) -> Decision | None:
    """Measurement support x class prior. None when fewer than two options can be weighed."""

    counts = PRIORS[kind]
    total = sum(counts.values())
    eps = FUSE_EPS.get(kind, 0.15)
    weights: dict[str, float] = {}
    for spec, sc in zip(program.options, scores):
        if spec.atoms is None or sc.verdict == "unverifiable":
            continue
        prior = (counts.get(class_key(spec.atoms), 0) + 0.5) / (total + 0.5 * max(len(counts), 1))
        weights[sc.letter] = (sc.score + eps) * prior
    if len(weights) < 2:
        return None
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    z = sum(weights.values())
    (letter, top), (_, second) = ranked[0], ranked[1]
    dup = {l for group in program.duplicates for l in group[1:] if group[0] == letter}
    if dup:
        rest = [w for l, w in ranked[1:] if l not in dup]
        second = rest[0] if rest else 0.0
    margin = (top - second) / z
    if result.confidence == "low":
        return Decision("B", letter, margin, scores, f"low-confidence measurement: {'; '.join(result.notes) or 'low'}", measured)
    if margin < DECISION["fused_margin"]:
        return Decision("B", letter, margin, scores, f"measurement and base rate leave two options close (posterior margin {margin:.2f})", measured)
    return Decision("A", letter, margin, scores,
                    f"option {letter} carries the posterior (margin {margin:.2f}) from the measurement and the base rate", measured)


def decide(program: QueryProgram, results: Mapping[str, Result],
           trusted: set[str] | None = None) -> Decision:
    kind = KIND_BY_NAME[program.kind]
    trusted = TRUSTED_KINDS if trusted is None else trusted
    if program.route == "vlm_only":
        return Decision("C", None, 0.0, [], "no measurement decides this question")
    result = main_result(program, results)
    scores = score_options(program, results)
    measured = {} if result is None else {"available": result.available, "confidence": result.confidence,
                                          "value": result.to_dict()["value"], "reason": result.reason}
    if result is None or not result.available:
        return Decision("B", None, 0.0, scores, f"measurement unavailable: {getattr(result, 'reason', 'none')}", measured)
    if not scores:
        return Decision("B", None, 0.0, scores, "the question has no options to score", measured)
    if kind.name in FUSE_KINDS and PRIORS.get(kind.name) and kind.name in trusted:
        fused = _fused_decision(program, result, scores, kind.name, measured)
        if fused is not None:
            return fused
    ranked = sorted(scores, key=lambda s: (-s.score, s.letter))
    top = ranked[0]
    second = ranked[1].score if len(ranked) > 1 else 0.0
    # duplicate options say one thing: the earlier letter is the answer and the later ones
    # are not "competition"
    dup = {l for group in program.duplicates for l in group[1:] if group[0] == top.letter}
    if dup:
        rest = [s.score for s in ranked[1:] if s.letter not in dup]
        second = rest[0] if rest else 0.0
    margin = top.score - second
    unverifiable = [s.letter for s in scores if s.verdict == "unverifiable"]
    if kind.name not in trusted:
        return Decision("B", top.letter, margin, scores, f"kind {kind.name!r} is not trusted for a deterministic answer yet", measured)
    if result.confidence == "low":
        return Decision("B", top.letter, margin, scores, f"low-confidence measurement: {'; '.join(result.notes) or result.confidence}", measured)
    if top.verdict == "unverifiable":
        return Decision("B", None, 0.0, scores, "the best-supported option makes a claim that cannot be checked", measured)
    g = gate(kind.name)
    if top.score < g["min_score"]:
        return Decision("B", top.letter, margin, scores, f"no option is well supported (best {top.score:.2f})", measured)
    if margin < g["min_margin"]:
        return Decision("B", top.letter, margin, scores, f"two options are close (margin {margin:.2f})", measured)
    if unverifiable and any(s.letter in unverifiable and s.score == 0 for s in scores) and program.answer_form == "order":
        return Decision("B", top.letter, margin, scores, "an event was not found", measured)
    return Decision("A", top.letter, margin, scores, f"option {top.letter} is supported ({top.score:.2f}) and clear of the rest (margin {margin:.2f})", measured)


def _who(program: QueryProgram, operand: str) -> str:
    if operand in ("cam", "camera"):
        return "the camera"
    base = operand.split("[")[0]
    entity = next((e for e in program.entities if e.id == base), None)
    if entity is None:
        return operand
    label = entity.phrase or "the main subject"
    return f"{label} #{int(operand.split('[')[1][:-1]) + 1}" if "[" in operand else label


def describe_result(program: QueryProgram, m: Any, r: Result) -> str:
    """One measurement as a plain sentence with its numbers, for a reader that is a language model."""

    v = r.value
    sig = m.signal
    if sig == "camera_motion":
        d = f" {v['direction']}" if v.get("direction") else ""
        return (f"the camera's dominant motion is {v['motion']}{d}. Net rotation: pan {v['pan_deg']:+.1f} deg "
                f"(+ is right), tilt {v['tilt_deg']:+.1f} deg (+ is up). Translation in scene scales: "
                f"{v['forward_over_scale']:+.2f} forward, {v['right_over_scale']:+.2f} right, {-v['down_over_scale']:+.2f} up.")
    if sig == "camera_jitter":
        names = ["stable", "slight shake", "shaky", "severe shake"]
        return f"camera shake is '{names[v['level']]}' (rotation jitter {v['rotation_residual_deg']:.2f} deg rms after removing the smooth path)."
    if sig in ("range", "region_distance"):
        a, b = (_who(program, o) for o in m.of[:2])
        if m.reduce == "value_at":
            metres = f"{v['metres']:.2f} m" if "metres" in v else f"{v['units']:.3f} model units (no metric scale)"
            return (f"the distance between {a} and {b} at the requested moment is about {metres}. Reliability: metric "
                    "depth is uncertain; expect this to be off by up to a factor of about two, so use it to rule out "
                    "options that are several times too large or too small, not to split close ones.")
        words = {"closer": "shrinks steadily", "farther": "grows steadily", "constant": "stays about the same",
                 "closer_then_farther": "first shrinks, then grows", "farther_then_closer": "first grows, then shrinks",
                 "fluctuating": "goes up and down repeatedly"}
        pl = f" (levels off from {v['plateau_from']:.0%} of the window)" if v.get("plateau_from") is not None else ""
        return (f"the distance between {a} and {b} {words.get(v['trend'], v['trend'])}{pl}: from {v['start_units']:.3f} to "
                f"{v.get('end_units', float('nan')):.3f} model units; changes under {v['tolerance_units']:.3f} are not counted.")
    if sig == "camera_displacement":
        d = direction_words(v)
        said = ", ".join(f"{name} {word}" for name, word in zip(("left/right", "up/down", "forward/backward"), d["words"].values()) if word)
        return (f"the camera moves {said or 'by too little to name a direction'} between frames {v['start']} and {v['end']} "
                f"(components in scene scales {np.round(np.array(d['components']) / v['scene_scale'], 2).tolist()}; +right, +down, +forward).")
    if sig == "displacement":
        d = direction_words(v)
        said = ", ".join(f"{name} {word}" for name, word in zip(("left/right", "up/down", "forward/backward"), d["words"].values()) if word)
        return (f"{_who(program, m.of[0])} moves {said or 'by too little to name a direction'} between frames {v['start']} and {v['end']} "
                f"(calibrated reading; components {np.round(d['components'], 3).tolist()}). Reliability: left/right is "
                "dependable; up/down less; forward/backward from a single camera's depth is close to a guess -- decide "
                "that part from the frames.")
    if sig == "instances":
        return (f"{v['count']} distinct instance(s) of {_who(program, m.of[0])} were found "
                f"({'stable' if v['stable'] else 'unstable'} across lifespan thresholds).")
    if sig == "speed_metric":
        who = _who(program, m.of[0])
        if "mps" not in v:
            return f"{who} moves at {v['units_per_s']:.3f} model units per second (no metric scale for this clip)."
        return (f"{who} moves at an average of about {v['mps']:.2f} m/s over {v['duration_s']:.1f} s (start-to-end "
                f"displacement {v['net_mps']:.2f} m/s; distance travelled {v['path_mps']:.2f} m/s; the camera's own motion is "
                "not counted). Reliability: metric depth is uncertain, so expect this to be off by up to a factor of about two; "
                "use it to rule out options several times too fast or too slow.")
    if sig == "onset":
        t = f" ({v['onset_s']:.1f} s)" if v.get("onset_s") is not None else ""
        return f"the event '{v['predicate']}' of {_who(program, v['actor'])} starts at frame {v['onset']}{t}." if v["onset"] is not None else f"the event '{v['predicate']}' was not found."
    if sig == "order":
        return f"order of the two events: {v['first']} (onsets at frames {v['onsets']})."
    if sig == "focal":
        return f"zoom: {v['zoom']} (focal length changes by x{v['focal_ratio_end']:.2f} from first to last frame; the camera also moved {v['forward_over_scale']:+.2f} scales forward)."
    if sig == "camera_speed":
        return f"camera speed profile: {v['rhythm'].replace('_', ' ')} (mean energy in the three thirds of the clip: {np.round(v['thirds'], 1).tolist()}; 1.0 is the moving threshold)."
    if sig == "covisibility":
        return f"{v['shared_fraction']:.0%} of the first view is seen again from the last camera position: the scene is '{v['revisit']}'."
    if sig == "area":
        return f"{_who(program, m.of[0])} covers {v['area_start']:.1%} of the image at first and {v['area_end']:.1%} at the end: it looks '{v['size']}'."
    if sig == "presence":
        return f"{_who(program, m.of[0])} is '{v['visible']}' in the asked window (present in {v['present_fraction']:.0%} of its frames)."
    if sig == "elevation":
        return f"the camera is '{v['angle']}' relative to {_who(program, m.of[0])} (median elevation {v['median_deg']:+.0f} deg above the horizon)."
    if sig == "speed":
        return f"{_who(program, m.of[0])} moves '{v['motion']}' ({v['moving_fraction']:.0%} of the time above the speed threshold, {v['runs']} run(s))."
    return str({k: v[k] for k in v if k not in ("pivots", "frames", "lifespans", "signed")})


def evidence_text(program: QueryProgram, results: Mapping[str, Result], decision: Decision) -> str:
    """What leaf B shows Qwen next to the frames: measurements, honestly labelled."""

    lines = ["A 3D reconstruction of the video was measured. Its findings (units are model-relative unless metres are stated):"]
    for m in program.measurements:
        r = results.get(m.id)
        if r is None:
            continue
        if not r.available:
            lines.append(f"- {m.signal}: could NOT be measured ({r.reason}).")
            continue
        lines.append(f"- [{r.confidence} confidence] {describe_result(program, m, r)}")
        lines.extend(f"    caution: {n}" for n in r.notes)
    if decision.scores and any(s.verdict != "unverifiable" for s in decision.scores):
        lines.append("Comparison of each option with those findings:")
        for s in decision.scores:
            lines.append(f"- {s.letter}: {s.verdict} -- {s.detail}")
    lines.append("The reconstruction can be wrong (segmentation or depth failures, thresholds); where it disagrees with the frames, trust what you can see.")
    return "\n".join(lines)
