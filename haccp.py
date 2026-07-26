#!/usr/bin/env python3
"""Reference HACCP evidence ledger for HTST simulator rows.

This module implements a deterministic, append-only *research evidence* layer.
It does not assess Korean, Codex or FDA conformity; does not validate a process;
does not authorize product release; and is not an electronic
signature implementation.  Wall-clock time is deliberately absent.  Every
record uses only the simulator row's explicit time and a canonical SHA-256 hash
chain so that repeated inputs produce byte-equivalent JSON-safe evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_HACCP_PLAN = HERE / "haccp_plan.json"
HACCP_PLAN_SCHEMA_VERSION = "1.0.0"
GENESIS_HASH = "0" * 64
MANDATORY_SOURCE_IDS = frozenset(
    {
        "KR-MFDS-2026-25-ART6",
        "CODEX-CXC-1-1969-2022",
        "FDA-NACMCF-HACCP-1997",
    }
)
REQUIRED_RUNTIME_SIGNALS = frozenset(
    {
        "safety_temp_sensor_c",
        "estimated_fastest_residence_time_s",
        "measured_flow_l_h",
        "measured_differential_pressure_bar",
        "cip_release_permissive",
    }
)
_LIMIT_OPERATORS = frozenset({">=", "<=", "=="})
_CONTROL_TYPES = frozenset({"CCP", "OPRP"})
_HAZARD_CATEGORIES = frozenset({"PROCESS_CONTROL", "CHEMICAL", "PHYSICAL"})
_REQUIRED_ACTIONS = frozenset({"HOLD", "DIVERT", "QUARANTINE"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class HACCPPlanError(ValueError):
    """Raised when the reference plan violates its strict data contract."""


class EvidenceError(ValueError):
    """Raised when runtime evidence cannot be represented deterministically."""


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"value is not canonical JSON-safe: {exc}") from exc
    return text.encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HACCPPlanError(f"{context} must be an object")
    return value


def _exact_keys(
    item: Mapping[object, object], required: set[str], context: str
) -> None:
    actual = set(item)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required, key=repr)
        raise HACCPPlanError(
            f"{context} keys mismatch; missing={missing}, extra={extra}"
        )


def _nonempty_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise HACCPPlanError(f"{context} must be a non-empty array")
    return value


def _text(item: Mapping[str, Any], name: str, context: str) -> str:
    value = item.get(name)
    if not isinstance(value, str) or not value.strip():
        raise HACCPPlanError(f"{context}.{name} must be a non-empty string")
    return value.strip()


def _identifier(item: Mapping[str, Any], name: str, context: str) -> str:
    value = _text(item, name, context)
    if not _IDENTIFIER.fullmatch(value):
        raise HACCPPlanError(f"{context}.{name} has invalid identifier syntax")
    return value


def _string_list(value: object, context: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise HACCPPlanError(f"{context} must be {qualifier}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise HACCPPlanError(f"{context} must contain non-empty strings")
    result = [item.strip() for item in value]
    if len(result) != len(set(result)):
        raise HACCPPlanError(f"{context} must not contain duplicates")
    return result


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HACCPPlanError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise HACCPPlanError(f"{context} must be finite")
    return number


def validate_haccp_plan(plan: Mapping[str, Any]) -> dict[str, object]:
    """Validate all seven-principle, hazard, control and record-plan fields."""

    root = _mapping(plan, "HACCP plan")
    required_top = {
        "schema_version",
        "plan_id",
        "title",
        "conformity_status",
        "model_status",
        "shadow_only",
        "assurance",
        "product_and_scope",
        "source_registry",
        "hazard_analysis",
        "controls",
    }
    _exact_keys(root, required_top, "HACCP plan")
    if root["schema_version"] != HACCP_PLAN_SCHEMA_VERSION:
        raise HACCPPlanError("unsupported HACCP plan schema_version")
    plan_id = _identifier(root, "plan_id", "HACCP plan")
    _text(root, "title", "HACCP plan")
    status = _text(root, "conformity_status", "HACCP plan")
    if status != "not_assessed":
        raise HACCPPlanError("conformity_status must remain 'not_assessed'")
    model_status = _text(root, "model_status", "HACCP plan").lower()
    for forbidden_claim in ("not a haccp certification", "not", "electronic signature"):
        if forbidden_claim not in model_status:
            raise HACCPPlanError(
                "model_status must explicitly disclaim certification, release and electronic signatures"
            )
    if root["shadow_only"] is not True:
        raise HACCPPlanError("HACCP plan must declare shadow_only=true")

    assurance = _mapping(root["assurance"], "assurance")
    _exact_keys(
        assurance,
        {
            "regulatory_conformity",
            "plant_validation",
            "process_safety_validation",
            "certification_claim",
            "electronic_signature_claim",
            "automatic_product_release",
        },
        "assurance",
    )
    for name in (
        "regulatory_conformity",
        "plant_validation",
        "process_safety_validation",
    ):
        if assurance.get(name) != "not_assessed":
            raise HACCPPlanError(f"assurance.{name} must be 'not_assessed'")
    for name in (
        "certification_claim",
        "electronic_signature_claim",
        "automatic_product_release",
    ):
        if assurance.get(name) is not False:
            raise HACCPPlanError(f"assurance.{name} must be false")

    product = _mapping(root["product_and_scope"], "product_and_scope")
    _exact_keys(
        product,
        {
            "product",
            "process",
            "intended_use",
            "consumer",
            "scope_start",
            "scope_end",
            "flow_diagram_status",
            "prerequisite_programs_status",
        },
        "product_and_scope",
    )
    for name in (
        "product",
        "process",
        "intended_use",
        "consumer",
        "scope_start",
        "scope_end",
        "flow_diagram_status",
        "prerequisite_programs_status",
    ):
        _text(product, name, "product_and_scope")

    sources = _nonempty_list(root["source_registry"], "source_registry")
    source_ids: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = _mapping(raw_source, f"source_registry[{index}]")
        context = f"source_registry[{index}]"
        _exact_keys(
            source,
            {
                "source_id",
                "authority",
                "title",
                "identifier",
                "effective_date",
                "provision",
                "url",
                "use",
                "conformity_status",
                "principles",
            },
            context,
        )
        source_id = _identifier(source, "source_id", context)
        if source_id in source_ids:
            raise HACCPPlanError(f"duplicate source_id {source_id!r}")
        source_ids.add(source_id)
        for name in (
            "authority",
            "title",
            "identifier",
            "effective_date",
            "provision",
            "url",
            "use",
        ):
            _text(source, name, context)
        if not str(source["url"]).startswith("https://"):
            raise HACCPPlanError(f"{context}.url must use https")
        if source.get("conformity_status") != "not_assessed":
            raise HACCPPlanError(f"{context}.conformity_status must be not_assessed")
        principles = _nonempty_list(source.get("principles"), f"{context}.principles")
        if len(principles) != 7:
            raise HACCPPlanError(f"{context}.principles must contain seven principles")
        numbers: list[int] = []
        for principle_index, raw_principle in enumerate(principles):
            principle = _mapping(
                raw_principle, f"{context}.principles[{principle_index}]"
            )
            _exact_keys(
                principle,
                {"number", "name"},
                f"{context}.principles[{principle_index}]",
            )
            number = principle.get("number")
            if isinstance(number, bool) or not isinstance(number, int):
                raise HACCPPlanError("principle number must be an integer")
            numbers.append(number)
            _text(principle, "name", f"{context}.principles[{principle_index}]")
        if numbers != list(range(1, 8)):
            raise HACCPPlanError(f"{context}.principles must be ordered 1 through 7")
    if not MANDATORY_SOURCE_IDS.issubset(source_ids):
        raise HACCPPlanError(
            f"source_registry lacks mandatory sources: {sorted(MANDATORY_SOURCE_IDS - source_ids)}"
        )
    korea = next(
        source for source in sources if source.get("source_id") == "KR-MFDS-2026-25-ART6"
    )
    if (
        "제2026-25호" not in str(korea.get("identifier"))
        or "제6조" not in str(korea.get("provision"))
    ):
        raise HACCPPlanError("Korean source must identify Notice 2026-25 Article 6")

    hazards = _nonempty_list(root["hazard_analysis"], "hazard_analysis")
    hazard_ids: set[str] = set()
    hazard_control_refs: dict[str, set[str]] = {}
    categories: set[str] = set()
    for index, raw_hazard in enumerate(hazards):
        hazard = _mapping(raw_hazard, f"hazard_analysis[{index}]")
        context = f"hazard_analysis[{index}]"
        _exact_keys(
            hazard,
            {
                "hazard_id",
                "category",
                "process_step",
                "hazard",
                "significance",
                "rationale",
                "control_ids",
                "source_ids",
                "scope_status",
            },
            context,
        )
        hazard_id = _identifier(hazard, "hazard_id", context)
        if hazard_id in hazard_ids:
            raise HACCPPlanError(f"duplicate hazard_id {hazard_id!r}")
        hazard_ids.add(hazard_id)
        category = _text(hazard, "category", context)
        if category not in _HAZARD_CATEGORIES:
            raise HACCPPlanError(f"{context}.category is unsupported")
        categories.add(category)
        for name in (
            "process_step",
            "hazard",
            "significance",
            "rationale",
            "scope_status",
        ):
            _text(hazard, name, context)
        controls = _string_list(hazard.get("control_ids"), f"{context}.control_ids", allow_empty=True)
        hazard_control_refs[hazard_id] = set(controls)
        referenced_sources = _string_list(hazard.get("source_ids"), f"{context}.source_ids")
        if not set(referenced_sources).issubset(source_ids):
            raise HACCPPlanError(f"{context} references an unknown source_id")
        if hazard["scope_status"] == "runtime_evaluated" and not controls:
            raise HACCPPlanError(f"{context} runtime hazard requires a control_id")
        if not controls and "outside_runtime_scope" not in str(hazard["scope_status"]):
            raise HACCPPlanError(f"{context} uncontrolled hazard must be outside runtime scope")
    if categories != set(_HAZARD_CATEGORIES):
        raise HACCPPlanError(
            "hazard_analysis must explicitly cover process-control, chemical and physical hazards"
        )

    controls = _nonempty_list(root["controls"], "controls")
    control_ids: set[str] = set()
    limit_ids: set[str] = set()
    runtime_signals: list[str] = []
    control_hazard_refs: dict[str, set[str]] = {}
    calibration_tags: set[str] = set()
    for index, raw_control in enumerate(controls):
        control = _mapping(raw_control, f"controls[{index}]")
        context = f"controls[{index}]"
        _exact_keys(
            control,
            {
                "control_id",
                "control_type",
                "process_step",
                "hazard_ids",
                "applicability",
                "critical_limits",
                "monitoring",
                "corrective_action",
                "verification",
                "records",
            },
            context,
        )
        control_id = _identifier(control, "control_id", context)
        if control_id in control_ids:
            raise HACCPPlanError(f"duplicate control_id {control_id!r}")
        control_ids.add(control_id)
        control_type = _text(control, "control_type", context)
        if control_type not in _CONTROL_TYPES:
            raise HACCPPlanError(f"{context}.control_type must be CCP or OPRP")
        _text(control, "process_step", context)
        if control.get("applicability") != "production_only":
            raise HACCPPlanError(f"{context}.applicability must be production_only")
        references = _string_list(control.get("hazard_ids"), f"{context}.hazard_ids")
        if not set(references).issubset(hazard_ids):
            raise HACCPPlanError(f"{context} references an unknown hazard_id")
        control_hazard_refs[control_id] = set(references)

        limits = _nonempty_list(control.get("critical_limits"), f"{context}.critical_limits")
        for limit_index, raw_limit in enumerate(limits):
            limit_item = _mapping(
                raw_limit, f"{context}.critical_limits[{limit_index}]"
            )
            limit_context = f"{context}.critical_limits[{limit_index}]"
            common_limit_keys = {
                "limit_id",
                "signal",
                "operator",
                "unit",
                "calibration_tags",
                "basis_type",
                "rationale",
                "source_ids",
            }
            actual_limit_keys = set(limit_item)
            variable_limit_keys = actual_limit_keys & {"value", "reference_signal"}
            if len(variable_limit_keys) != 1:
                raise HACCPPlanError(
                    f"{limit_context} must define exactly one of value/reference_signal"
                )
            _exact_keys(
                limit_item,
                common_limit_keys | variable_limit_keys,
                limit_context,
            )
            limit_id = _identifier(limit_item, "limit_id", limit_context)
            if limit_id in limit_ids:
                raise HACCPPlanError(f"duplicate limit_id {limit_id!r}")
            limit_ids.add(limit_id)
            signal = _text(limit_item, "signal", limit_context)
            runtime_signals.append(signal)
            operator = _text(limit_item, "operator", limit_context)
            if operator not in _LIMIT_OPERATORS:
                raise HACCPPlanError(f"{limit_context}.operator is unsupported")
            has_value = "value" in limit_item
            has_reference = "reference_signal" in limit_item
            if has_value == has_reference:
                raise HACCPPlanError(
                    f"{limit_context} must define exactly one of value/reference_signal"
                )
            if has_value:
                _finite(limit_item["value"], f"{limit_context}.value")
            else:
                reference_signal = _text(limit_item, "reference_signal", limit_context)
                if reference_signal in REQUIRED_RUNTIME_SIGNALS:
                    raise HACCPPlanError(
                        f"{limit_context}.reference_signal must be a separate limit signal"
                    )
            for name in ("unit", "basis_type", "rationale"):
                _text(limit_item, name, limit_context)
            if "not_validated" not in str(limit_item["basis_type"]):
                raise HACCPPlanError(f"{limit_context}.basis_type must state not_validated")
            tags = _string_list(
                limit_item.get("calibration_tags"), f"{limit_context}.calibration_tags"
            )
            for tag in tags:
                if not _IDENTIFIER.fullmatch(tag):
                    raise HACCPPlanError(f"{limit_context} has invalid calibration tag")
            calibration_tags.update(tags)
            referenced_sources = _string_list(
                limit_item.get("source_ids"), f"{limit_context}.source_ids"
            )
            if not set(referenced_sources).issubset(source_ids):
                raise HACCPPlanError(f"{limit_context} references an unknown source_id")

        monitoring = _mapping(control.get("monitoring"), f"{context}.monitoring")
        _exact_keys(
            monitoring,
            {"what", "how", "frequency", "who"},
            f"{context}.monitoring",
        )
        for name in ("what", "how", "frequency", "who"):
            _text(monitoring, name, f"{context}.monitoring")
        corrective = _mapping(
            control.get("corrective_action"), f"{context}.corrective_action"
        )
        _exact_keys(
            corrective,
            {
                "immediate_actions",
                "investigation",
                "disposition",
                "release_authority",
                "automatic_release",
            },
            f"{context}.corrective_action",
        )
        actions = _string_list(
            corrective.get("immediate_actions"),
            f"{context}.corrective_action.immediate_actions",
        )
        if not _REQUIRED_ACTIONS.issubset(actions):
            raise HACCPPlanError(
                f"{context} corrective actions must include HOLD, DIVERT and QUARANTINE"
            )
        for name in ("investigation", "disposition", "release_authority"):
            _text(corrective, name, f"{context}.corrective_action")
        if corrective.get("automatic_release") is not False:
            raise HACCPPlanError(f"{context} must forbid automatic release")
        verification = _mapping(control.get("verification"), f"{context}.verification")
        _exact_keys(
            verification,
            {"activities", "frequency", "who"},
            f"{context}.verification",
        )
        _string_list(
            verification.get("activities"), f"{context}.verification.activities"
        )
        for name in ("frequency", "who"):
            _text(verification, name, f"{context}.verification")
        records = _string_list(control.get("records"), f"{context}.records")
        required_records = {
            "MONITORING_EVIDENCE",
            "DEVIATION_RECORD",
            "CONTAINMENT_ACTION",
            "CALIBRATION_STATUS_REFERENCE",
            "HASH_CHAIN_VERIFICATION",
        }
        if not required_records.issubset(records):
            raise HACCPPlanError(f"{context}.records is missing required evidence types")

    if set(runtime_signals) != set(REQUIRED_RUNTIME_SIGNALS):
        raise HACCPPlanError(
            "runtime critical limits must cover exactly the five required signals: "
            f"missing={sorted(REQUIRED_RUNTIME_SIGNALS - set(runtime_signals))}, "
            f"extra={sorted(set(runtime_signals) - REQUIRED_RUNTIME_SIGNALS)}"
        )
    if len(runtime_signals) != len(set(runtime_signals)):
        raise HACCPPlanError("each required runtime signal must have exactly one critical limit")
    for hazard_id, references in hazard_control_refs.items():
        unknown = references - control_ids
        if unknown:
            raise HACCPPlanError(f"hazard {hazard_id!r} references unknown controls {sorted(unknown)}")
        for control_id in references:
            if hazard_id not in control_hazard_refs.get(control_id, set()):
                raise HACCPPlanError(
                    f"hazard/control references are not bidirectional: {hazard_id}/{control_id}"
                )
    for control_id, references in control_hazard_refs.items():
        for hazard_id in references:
            if control_id not in hazard_control_refs.get(hazard_id, set()):
                raise HACCPPlanError(
                    f"control/hazard references are not bidirectional: {control_id}/{hazard_id}"
                )

    return {
        "valid": True,
        "plan_id": plan_id,
        "schema_version": HACCP_PLAN_SCHEMA_VERSION,
        "conformity_status": "not_assessed",
        "shadow_only": True,
        "source_count": len(source_ids),
        "hazard_count": len(hazard_ids),
        "control_count": len(control_ids),
        "critical_limit_count": len(limit_ids),
        "calibration_tag_count": len(calibration_tags),
        "runtime_signals": sorted(runtime_signals),
    }


def _reject_json_constant(value: str) -> None:
    raise HACCPPlanError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HACCPPlanError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_haccp_plan(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the bundled plan or an explicit JSON plan."""

    source = DEFAULT_HACCP_PLAN if path is None else Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except HACCPPlanError:
        raise
    except OSError as exc:
        raise HACCPPlanError(f"cannot read HACCP plan {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HACCPPlanError(f"invalid HACCP plan JSON {source}: {exc}") from exc
    plan = dict(_mapping(value, "HACCP plan"))
    validate_haccp_plan(plan)
    return plan


def _runtime_number(row: Mapping[str, object], name: str) -> tuple[float | None, str | None]:
    if name not in row:
        return None, f"missing_signal:{name}"
    value = row[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"non_numeric_signal:{name}"
    number = float(value)
    if not math.isfinite(number):
        return None, f"nonfinite_signal:{name}"
    return number, None


def _runtime_bool(row: Mapping[str, object], name: str) -> tuple[bool | None, str | None]:
    if name in row and isinstance(row[name], bool):
        return row[name], None
    number, error = _runtime_number(row, name)
    if error is not None:
        return None, error
    if number not in {0.0, 1.0}:
        return None, f"non_boolean_signal:{name}"
    return bool(int(number)), None


def _compare(observed: float, operator: str, limit: float) -> bool:
    if operator == ">=":
        return observed >= limit
    if operator == "<=":
        return observed <= limit
    if operator == "==":
        return observed == limit
    raise EvidenceError(f"unsupported runtime operator {operator!r}")


def _calibration_check(
    tags: Sequence[str],
    statuses: Mapping[str, object] | None,
    time_s: float,
) -> tuple[str, list[dict[str, object]]]:
    details: list[dict[str, object]] = []
    if statuses is None:
        return "UNKNOWN", [{"tag": tag, "status": "MISSING"} for tag in tags]
    if not isinstance(statuses, Mapping):
        return "UNKNOWN", [{"tag": tag, "status": "INVALID_CONTAINER"} for tag in tags]
    has_expired = False
    has_unknown = False
    for tag in tags:
        raw = statuses.get(tag)
        valid_until: float | None = None
        if isinstance(raw, str):
            status = raw.upper()
        elif isinstance(raw, Mapping):
            if set(raw) not in ({"status"}, {"status", "valid_until_time_s"}):
                status = "UNKNOWN"
                details.append({"tag": tag, "status": status})
                has_unknown = True
                continue
            raw_status = raw.get("status")
            status = raw_status.upper() if isinstance(raw_status, str) else "UNKNOWN"
            if "valid_until_time_s" in raw:
                value = raw["valid_until_time_s"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    status = "UNKNOWN"
                else:
                    valid_until = float(value)
        else:
            status = "MISSING"
        if status not in {"VALID", "EXPIRED", "UNKNOWN", "MISSING"}:
            status = "UNKNOWN"
        if status == "VALID" and valid_until is not None and time_s >= valid_until:
            status = "EXPIRED"
        detail: dict[str, object] = {"tag": tag, "status": status}
        if valid_until is not None:
            detail["valid_until_time_s"] = valid_until
        details.append(detail)
        has_expired = has_expired or status == "EXPIRED"
        has_unknown = has_unknown or status in {"UNKNOWN", "MISSING"}
    if has_expired:
        return "EXPIRED", details
    if has_unknown:
        return "UNKNOWN", details
    return "VALID", details


def verify_evidence_chain(
    records: Sequence[Mapping[str, object]],
    *,
    expected_record_count: int | None = None,
    expected_chain_head: str | None = None,
) -> dict[str, object]:
    """Recompute the chain and optionally compare it with an external anchor."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        return {"valid": False, "record_count": 0, "error": "records_not_a_sequence"}
    if (
        expected_record_count is not None
        and (
            isinstance(expected_record_count, bool)
            or not isinstance(expected_record_count, int)
            or expected_record_count < 0
        )
    ):
        return {
            "valid": False,
            "record_count": len(records),
            "error": "invalid_expected_record_count",
        }
    if expected_chain_head is not None and (
        not isinstance(expected_chain_head, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_chain_head) is None
    ):
        return {
            "valid": False,
            "record_count": len(records),
            "error": "invalid_expected_chain_head",
        }
    previous = GENESIS_HASH
    for position, raw in enumerate(records, start=1):
        if not isinstance(raw, Mapping):
            return {
                "valid": False,
                "record_count": len(records),
                "error": f"record_{position}_not_an_object",
            }
        record = dict(raw)
        supplied_hash = record.pop("record_hash", None)
        if record.get("sequence") != position:
            return {
                "valid": False,
                "record_count": len(records),
                "error": f"record_{position}_sequence_mismatch",
            }
        if record.get("previous_hash") != previous:
            return {
                "valid": False,
                "record_count": len(records),
                "error": f"record_{position}_previous_hash_mismatch",
            }
        if record.get("record_type") == "HASH_CHAIN_VERIFICATION":
            if record.get("verified_record_count") != position - 1:
                return {
                    "valid": False,
                    "record_count": len(records),
                    "error": f"record_{position}_verification_count_mismatch",
                }
            if record.get("verified_chain_head") != previous:
                return {
                    "valid": False,
                    "record_count": len(records),
                    "error": f"record_{position}_verification_head_mismatch",
                }
        try:
            expected = _sha256(record)
        except EvidenceError:
            return {
                "valid": False,
                "record_count": len(records),
                "error": f"record_{position}_not_canonical_json",
            }
        if supplied_hash != expected:
            return {
                "valid": False,
                "record_count": len(records),
                "error": f"record_{position}_hash_mismatch",
            }
        previous = expected
    if expected_record_count is not None and len(records) != expected_record_count:
        return {
            "valid": False,
            "record_count": len(records),
            "chain_head": previous,
            "error": "external_record_count_mismatch",
        }
    if expected_chain_head is not None and previous != expected_chain_head:
        return {
            "valid": False,
            "record_count": len(records),
            "chain_head": previous,
            "error": "external_chain_head_mismatch",
        }
    return {
        "valid": True,
        "record_count": len(records),
        "chain_head": previous,
        "error": None,
    }


class EvidenceLedger:
    """Append-only deterministic evidence and product-disposition ledger."""

    def __init__(self, plan: Mapping[str, Any] | None = None) -> None:
        source = load_haccp_plan() if plan is None else copy.deepcopy(dict(plan))
        validate_haccp_plan(source)
        self._plan = source
        self.plan_sha256 = _sha256(source)
        self._records: list[dict[str, object]] = []
        self._lot_statuses: dict[str, str] = {}
        self._limits: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for control in self._plan["controls"]:
            for limit_item in control["critical_limits"]:
                self._limits.append((control, limit_item))

    @property
    def plan(self) -> dict[str, Any]:
        """Return a detached plan so runtime limits cannot diverge from its hash."""

        return copy.deepcopy(self._plan)

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        """Return detached record copies so callers cannot mutate the ledger."""

        return tuple(copy.deepcopy(self._records))

    @property
    def lot_statuses(self) -> dict[str, str]:
        return dict(self._lot_statuses)

    @property
    def chain_head(self) -> str:
        return self._records[-1]["record_hash"] if self._records else GENESIS_HASH

    def _append(self, payload: Mapping[str, object]) -> dict[str, object]:
        record = {
            "sequence": len(self._records) + 1,
            "record_id": f"HACCP-EVIDENCE-{len(self._records) + 1:08d}",
            "previous_hash": self.chain_head,
            **copy.deepcopy(dict(payload)),
        }
        if "record_hash" in record:
            raise EvidenceError("payload may not supply record_hash")
        record_hash = _sha256(record)
        completed = {**record, "record_hash": record_hash}
        self._records.append(completed)
        return copy.deepcopy(completed)

    @staticmethod
    def _lot_id(value: object) -> str:
        if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
            raise EvidenceError("lot_id must be a non-empty deterministic identifier")
        return value

    @staticmethod
    def _time(row: Mapping[str, object]) -> float:
        name = "campaign_time_s" if "campaign_time_s" in row else "time_s"
        if name not in row:
            raise EvidenceError("row must contain campaign_time_s or time_s")
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceError(f"{name} must be numeric")
        time_s = float(value)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise EvidenceError(f"{name} must be finite and non-negative")
        return time_s

    def evaluate(
        self,
        row: Mapping[str, object],
        calibration_statuses: Mapping[str, object] | None = None,
        lot_id: str = "REFERENCE-LOT",
    ) -> dict[str, object]:
        """Evaluate five reference controls and append canonical evidence.

        Missing or non-finite measurements and unknown calibration never pass.
        Expired calibration is a deviation.  Any deviation or unknown condition
        places the synthetic lot on HOLD and records conservative containment.
        Later passing rows cannot change HOLD to RELEASE.
        """

        if not isinstance(row, Mapping):
            raise EvidenceError("row must be a mapping")
        resolved_lot = self._lot_id(lot_id)
        time_s = self._time(row)
        start_index = len(self._records)
        self._lot_statuses.setdefault(resolved_lot, "UNREVIEWED")
        cip_active, context_error = _runtime_bool(row, "cip_cycle_active")
        evaluation_statuses: list[str] = []

        for control, limit_item in self._limits:
            signal = str(limit_item["signal"])
            observed: float | None = None
            limit_value: float | None = None
            calibration_status = "NOT_EVALUATED"
            calibration_details: list[dict[str, object]] = []
            required_actions = list(control["corrective_action"]["immediate_actions"])

            if context_error is not None:
                evidence_status = "UNKNOWN"
                reason = context_error
            elif cip_active:
                evidence_status = "NOT_APPLICABLE"
                reason = "control_applicability:production_only"
            else:
                observed, measurement_error = _runtime_number(row, signal)
                if "value" in limit_item:
                    limit_value = float(limit_item["value"])
                    limit_error = None
                else:
                    reference_name = str(limit_item["reference_signal"])
                    limit_value, limit_error = _runtime_number(row, reference_name)
                calibration_status, calibration_details = _calibration_check(
                    list(limit_item["calibration_tags"]),
                    calibration_statuses,
                    time_s,
                )
                if measurement_error is not None:
                    evidence_status = "UNKNOWN"
                    reason = measurement_error
                elif limit_error is not None:
                    evidence_status = "UNKNOWN"
                    reason = limit_error
                elif calibration_status == "EXPIRED":
                    evidence_status = "UNKNOWN"
                    reason = "expired_calibration"
                elif calibration_status != "VALID":
                    evidence_status = "UNKNOWN"
                    reason = "missing_or_unknown_calibration"
                elif observed is None or limit_value is None:  # defensive typing guard
                    evidence_status = "UNKNOWN"
                    reason = "unavailable_measurement_or_limit"
                elif _compare(observed, str(limit_item["operator"]), limit_value):
                    evidence_status = "PASS"
                    reason = "critical_limit_satisfied"
                else:
                    evidence_status = "DEVIATION"
                    reason = "critical_limit_failed"

            evaluation_statuses.append(evidence_status)
            if evidence_status in {"DEVIATION", "UNKNOWN"}:
                self._lot_statuses[resolved_lot] = "HOLD"
            calibration_reference = self._append(
                {
                    "record_type": "CALIBRATION_STATUS_REFERENCE",
                    "plan_id": self._plan["plan_id"],
                    "plan_sha256": self.plan_sha256,
                    "conformity_status": "not_assessed",
                    "lot_id": resolved_lot,
                    "time_s": time_s,
                    "control_id": control["control_id"],
                    "limit_id": limit_item["limit_id"],
                    "calibration_status": calibration_status,
                    "calibration_details": calibration_details,
                    "automatic_release_performed": False,
                }
            )
            monitoring = self._append(
                {
                    "record_type": "MONITORING_EVIDENCE",
                    "plan_id": self._plan["plan_id"],
                    "plan_sha256": self.plan_sha256,
                    "conformity_status": "not_assessed",
                    "lot_id": resolved_lot,
                    "time_s": time_s,
                    "control_id": control["control_id"],
                    "control_type": control["control_type"],
                    "limit_id": limit_item["limit_id"],
                    "signal": signal,
                    "operator": limit_item["operator"],
                    "observed_value": observed,
                    "limit_value": limit_value,
                    "evidence_status": evidence_status,
                    "reason": reason,
                    "calibration_status": calibration_status,
                    "calibration_details": calibration_details,
                    "calibration_reference_record_id": calibration_reference[
                        "record_id"
                    ],
                    "required_actions": (
                        required_actions
                        if evidence_status in {"DEVIATION", "UNKNOWN"}
                        else []
                    ),
                    "lot_status_after": self._lot_statuses[resolved_lot],
                    "automatic_release_performed": False,
                }
            )
            if evidence_status == "DEVIATION":
                self._append(
                    {
                        "record_type": "DEVIATION_RECORD",
                        "plan_id": self._plan["plan_id"],
                        "plan_sha256": self.plan_sha256,
                        "conformity_status": "not_assessed",
                        "lot_id": resolved_lot,
                        "time_s": time_s,
                        "control_id": control["control_id"],
                        "limit_id": limit_item["limit_id"],
                        "monitoring_record_id": monitoring["record_id"],
                        "reason": reason,
                        "lot_status_after": "HOLD",
                        "automatic_release_performed": False,
                    }
                )
            if evidence_status in {"DEVIATION", "UNKNOWN"}:
                for action in required_actions:
                    self._append(
                        {
                            "record_type": "CONTAINMENT_ACTION",
                            "plan_id": self._plan["plan_id"],
                            "plan_sha256": self.plan_sha256,
                            "conformity_status": "not_assessed",
                            "lot_id": resolved_lot,
                            "time_s": time_s,
                            "control_id": control["control_id"],
                            "limit_id": limit_item["limit_id"],
                            "monitoring_record_id": monitoring["record_id"],
                            "trigger_status": evidence_status,
                            "action": action,
                            "action_status": "RECORDED_REFERENCE_ACTION_NOT_PHYSICALLY_EXECUTED",
                            "lot_status_after": "HOLD",
                            "automatic_release_performed": False,
                        }
                    )

        active = [status for status in evaluation_statuses if status != "NOT_APPLICABLE"]
        if "UNKNOWN" in active:
            overall_status = "UNKNOWN"
        elif "DEVIATION" in active:
            overall_status = "DEVIATION"
        elif active and all(status == "PASS" for status in active):
            overall_status = "PASS"
        else:
            overall_status = "NOT_APPLICABLE"
        if (
            overall_status == "PASS"
            and self._lot_statuses[resolved_lot] != "HOLD"
        ):
            self._lot_statuses[resolved_lot] = "ELIGIBLE_FOR_MANUAL_REVIEW"
        counts = Counter(evaluation_statuses)
        summary_record = self._append(
            {
                "record_type": "EVALUATION_SUMMARY",
                "plan_id": self._plan["plan_id"],
                "plan_sha256": self.plan_sha256,
                "conformity_status": "not_assessed",
                "lot_id": resolved_lot,
                "time_s": time_s,
                "overall_status": overall_status,
                "status_counts": dict(sorted(counts.items())),
                "lot_status_after": self._lot_statuses[resolved_lot],
                "manual_release_required": True,
                "automatic_release_performed": False,
                "certification_claim": False,
                "electronic_signature_claim": False,
            }
        )
        prior_verification = self.verify()
        if not prior_verification["valid"]:
            raise EvidenceError(
                f"cannot append verification record to invalid chain: {prior_verification}"
            )
        verification_record = self._append(
            {
                "record_type": "HASH_CHAIN_VERIFICATION",
                "plan_id": self._plan["plan_id"],
                "plan_sha256": self.plan_sha256,
                "conformity_status": "not_assessed",
                "lot_id": resolved_lot,
                "time_s": time_s,
                "verified_record_count": prior_verification["record_count"],
                "verified_chain_head": prior_verification["chain_head"],
                "verification_method": "canonical_json_sha256_internal_recomputation",
                "external_anchor_present": False,
                "certification_claim": False,
                "electronic_signature_claim": False,
                "automatic_release_performed": False,
            }
        )
        appended = copy.deepcopy(self._records[start_index:])
        return {
            "plan_id": self._plan["plan_id"],
            "conformity_status": "not_assessed",
            "shadow_only": True,
            "lot_id": resolved_lot,
            "time_s": time_s,
            "overall_status": overall_status,
            "status_counts": dict(sorted(counts.items())),
            "lot_status": self._lot_statuses[resolved_lot],
            "manual_release_required": True,
            "automatic_release_performed": False,
            "summary_record_id": summary_record["record_id"],
            "chain_verification_record_id": verification_record["record_id"],
            "records_appended": appended,
            "chain_head": self.chain_head,
        }

    def verify(self) -> dict[str, object]:
        return verify_evidence_chain(self._records)

    def export_json_safe(self) -> dict[str, object]:
        verification = self.verify()
        if not verification["valid"]:
            raise EvidenceError(f"cannot export invalid evidence chain: {verification}")
        result = {
            "schema_version": "1.0.0",
            "artifact_type": "reference_haccp_evidence",
            "plan_id": self._plan["plan_id"],
            "plan_sha256": self.plan_sha256,
            "conformity_status": "not_assessed",
            "shadow_only": True,
            "certification_claim": False,
            "electronic_signature_claim": False,
            "automatic_product_release": False,
            "lot_statuses": dict(sorted(self._lot_statuses.items())),
            "record_count": len(self._records),
            "chain_head": self.chain_head,
            "chain_verification": verification,
            "records": copy.deepcopy(self._records),
        }
        _canonical_bytes(result)
        return result

    def export_csv_safe(self) -> dict[str, object]:
        """Return fixed-column rows whose nested values are canonical JSON text."""

        verification = self.verify()
        if not verification["valid"]:
            raise EvidenceError(f"cannot export invalid evidence chain: {verification}")
        rows: list[dict[str, object]] = []
        keys: set[str] = set()
        normalized_records: list[dict[str, object]] = []
        for record in self._records:
            normalized: dict[str, object] = {}
            for name, value in record.items():
                if isinstance(value, (dict, list, tuple)):
                    normalized[name] = _canonical_bytes(value).decode("utf-8")
                elif value is None:
                    normalized[name] = ""
                elif isinstance(value, (str, int, float, bool)):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise EvidenceError("non-finite value cannot be exported to CSV")
                    normalized[name] = value
                else:
                    raise EvidenceError(f"unsupported CSV value type for {name}")
            keys.update(normalized)
            normalized_records.append(normalized)
        preferred = [
            "sequence",
            "record_id",
            "record_type",
            "plan_id",
            "lot_id",
            "time_s",
            "control_id",
            "limit_id",
            "evidence_status",
            "overall_status",
            "action",
            "reason",
            "lot_status_after",
            "previous_hash",
            "record_hash",
        ]
        fieldnames = [name for name in preferred if name in keys]
        fieldnames.extend(sorted(keys - set(fieldnames)))
        for normalized in normalized_records:
            rows.append({name: normalized.get(name, "") for name in fieldnames})
        return {"fieldnames": fieldnames, "rows": rows}


__all__ = [
    "DEFAULT_HACCP_PLAN",
    "EvidenceError",
    "EvidenceLedger",
    "GENESIS_HASH",
    "HACCPPlanError",
    "HACCP_PLAN_SCHEMA_VERSION",
    "MANDATORY_SOURCE_IDS",
    "REQUIRED_RUNTIME_SIGNALS",
    "load_haccp_plan",
    "validate_haccp_plan",
    "verify_evidence_chain",
]
