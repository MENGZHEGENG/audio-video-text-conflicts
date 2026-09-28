from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

import conflictbench.perception_media_edits as media_edits
import conflictbench.perception_omni_gate as omni_gate
from conflictbench.perception_media_edits import (
    MediaEditValidationError,
    _ffmpeg_arguments,
    _probe_media_profile,
    build_edit_plan,
    construct_controlled_media,
    verify_controlled_media,
)
from conflictbench.perception_media_pilot import implementation_provenance
from conflictbench.perception_nuisance_gate import (
    IMPLEMENTATION_SOURCE_ROLES,
    build_nuisance_contract,
)

RUNNER = (
    pathlib.Path(__file__).parents[1] / "scripts" / "run_perception_controlled_media.py"
)


@pytest.fixture
def stub_source_gate_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep controlled-media tests focused on the verifier boundary.

    The source-gate module has its own replay integration suite. These tests
    exercise the controlled-media consumer with a deterministic replay result.
    """

    def replay(output: dict, **_kwargs: object) -> dict:
        return output

    monkeypatch.setattr(media_edits, "verify_gate_output_replay", replay, raising=False)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _attest(value: dict) -> dict:
    result = dict(value)
    result["attestation_sha256"] = _canonical_digest(result)
    return result


def _file_digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configuration() -> dict:
    return {
        "schema_version": 4,
        "runner": "perception_test_controlled_media_edits",
        "conditions": [
            "clean_original",
            "remux_only",
            "same_answer",
            "opposite_answer",
            "missing_source",
            "temporal_shift",
        ],
        "requested_source_orientations": [
            "audio_over_video",
            "video_over_audio",
        ],
        "source_gate_authorization": {
            "verification": "exact_output_from_authenticated_transcript_replay",
            "required_source_sufficiency_status": "pass",
            "required_nuisance_detection_status": "not_run",
            "required_overall_pilot_status": "pending_nuisance_detection",
            "orientation_conditions": {
                "audio_over_video": "audio_only",
                "video_over_audio": "video_only",
            },
        },
        "missing_source_fill": {"audio": "silence", "video": "black"},
        "temporal_calibration": {
            "record_schema": "conflictbench.perception-temporal-shift-events.v2",
            "replay_protocol": "pinned_python_json_cli_v1",
            "selection_rule": (
                "maximum_mean_absolute_score_change_then_smallest_shift_seconds"
            ),
            "event_scoring_rule": "absolute_reference_minus_shifted",
            "implementation_roles": ["generator", "scorer"],
            "candidate_shifts_seconds": [0.25, 0.5, 1.0],
            "official_split": "train",
            "calibration_partition": "scorer_fit",
        },
        "audio_encoding": {
            "codec": "aac",
            "bitrate": "128k",
            "bitrate_tolerance_fraction": 0.35,
            "sample_rate": 48000,
            "channels": 2,
            "sample_format": "fltp",
        },
        "video_encoding": {
            "codec": "h264",
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "crf": 18,
        },
        "container": {
            "format": "mp4",
            "probe_format_names": ["mov", "mp4", "m4a", "3gp", "3g2", "mj2"],
            "strip_metadata": True,
        },
        "validation": {
            "duration_tolerance_seconds": 0.1,
            "ffmpeg_timeout_seconds": 30,
            "require_exact_stream_types": ["audio", "video"],
            "reject_extra_streams": True,
        },
        "human_evaluation": "forbidden",
    }


def _build_plan(pilot: dict) -> list[dict]:
    return build_edit_plan(
        pilot,
        _normalized_media(pilot),
        _configuration(),
        selected_shift_seconds=0.5,
        authorized_orientations=("audio_over_video", "video_over_audio"),
    )


def _nuisance_configuration() -> dict:
    from conflictbench.perception_nuisance_gate import FEATURE_NAMES

    return {
        "schema_version": 1,
        "runner": "perception_edit_role_nuisance_detector",
        "feature_set": {
            "name": "container_and_stream_diagnostics_v1",
            "names": list(FEATURE_NAMES),
            "forbidden_inputs": [
                "question",
                "options",
                "answer",
                "semantic_embedding",
                "source_answer_score",
                "source_sufficiency_outcome",
            ],
        },
        "model": {
            "families": [
                "ridge_linear",
                "univariate_linear",
                "pairwise_interaction",
            ],
            "ridge_l2": 1.0,
            "selection": "fit_only_max_absolute_covariance",
        },
        "protocol": {
            "fit_partition": "scorer_fit",
            "threshold_partition": "threshold_calibration",
            "evaluation_partition": "pilot_gate",
            "positive_role": "opposite_answer_candidate",
            "negative_role": "same_answer_nuisance",
            "threshold_selection": ("maximum_balanced_accuracy_then_closest_to_zero"),
            "question_key_sensitivity": ("evaluation_keys_unseen_in_fit_or_threshold"),
            "minimum_unseen_question_components": 2,
            "human_evaluation": "forbidden",
            "outcome_based_sample_selection": "forbidden",
        },
        "uncertainty": {
            "method": "component_cluster_percentile_wilson_and_max_statistic",
            "confidence_level": 0.95,
            "bootstrap_repetitions": 32,
            "bootstrap_seed": 2301,
        },
        "permutation": {
            "method": "within_component_role_swap_with_refit",
            "repetitions": 32,
            "seed": 4129,
        },
        "gate": {
            "maximum_primary_upper_confidence_bound": 1.0,
            "maximum_unseen_question_upper_confidence_bound": 1.0,
            "maximum_point_lift_over_structural_chance": 1.0,
            "maximum_structural_chance_balanced_accuracy": 1.0,
            "minimum_permutation_p_value_exclusive": 0.0,
        },
    }


def _pilot() -> dict:
    partitions = [
        "scorer_fit",
        "scorer_fit",
        "threshold_calibration",
        "threshold_calibration",
        "pilot_gate",
        "pilot_gate",
    ]
    pairs = []
    components = []
    next_video = 1
    for target_index, partition in enumerate(partitions):
        target_video_id = f"video_{next_video:04d}"
        same_video_id = f"video_{next_video + 1:04d}"
        opposite_video_id = f"video_{next_video + 2:04d}"
        next_video += 3
        question = f"What happened in example {target_index}?"
        options = ["alpha", "beta", "unused"]
        answer_id = target_index % 2
        answer = options[answer_id]
        opposite_id = 1 - answer_id
        target = {
            "video_id": target_video_id,
            "question_id": 0,
            "question": question,
            "options": options,
            "answer_id": answer_id,
            "answer": answer,
        }
        anchor = {
            "question": question,
            "options": options,
            "answer_id": answer_id,
            "answer": answer,
        }
        component_id = f"{target_index + 1:024x}"
        key_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        for role, donor_video_id, donor_answer_id in (
            ("same_answer_nuisance", same_video_id, answer_id),
            ("opposite_answer_candidate", opposite_video_id, opposite_id),
        ):
            donor = {
                "video_id": donor_video_id,
                "question_id": 0,
                "question": question,
                "options": options,
                "answer_id": donor_answer_id,
                "answer": options[donor_answer_id],
            }
            identity = f"{target_video_id}:{role}".encode()
            pairs.append(
                {
                    "pair_id": hashlib.sha256(identity).hexdigest()[:24],
                    "component_id": component_id,
                    "official_split": "train",
                    "partition": partition,
                    "key_sha256": key_sha256,
                    "role": role,
                    "anchor": anchor,
                    "target": target,
                    "donor": donor,
                }
            )
        components.append(
            {
                "component_id": component_id,
                "partition": partition,
                "target_count": 1,
                "video_ids": [target_video_id, same_video_id, opposite_video_id],
            }
        )
    provenance = implementation_provenance()
    pilot = {
        "schema_version": 1,
        "implementation_sha256": provenance["implementation_sha256"],
        "implementation_source_sha256": provenance["source_sha256"],
        "candidate_index": sorted(
            pairs,
            key=lambda item: (
                item["partition"],
                item["component_id"],
                item["pair_id"],
            ),
        ),
        "components": sorted(
            components, key=lambda item: (item["partition"], item["component_id"])
        ),
    }
    pilot["attestation_sha256"] = _canonical_digest(pilot)
    return pilot


def _normalized_media(pilot: dict) -> list[dict]:
    video_ids = sorted(
        {
            source["video_id"]
            for pair in pilot["candidate_index"]
            for source in (pair["target"], pair["donor"])
        }
    )
    return [
        {
            "video_id": video_id,
            "filename": f"{video_id}.mp4",
            "size_bytes": 10,
            "sha256": hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
            "stream_types": ["audio", "video"],
        }
        for video_id in video_ids
    ]


def test_edit_plan_covers_five_controls_and_preserves_target_anchors() -> None:
    pilot = _pilot()

    plan = _build_plan(pilot)

    assert len(plan) == 60
    target_groups: dict[str, list[dict]] = {}
    for edit in plan:
        target_groups.setdefault(edit["target_id"], []).append(edit)
        assert edit["anchor"] == {
            "question": edit["target"]["question"],
            "options": edit["target"]["options"],
            "answer_id": edit["target"]["answer_id"],
            "answer": edit["target"]["answer"],
        }
        assert edit["source_files"][0]["video_id"] == edit["target_video_id"]
    assert {len(edits) for edits in target_groups.values()} == {10}
    assert {
        edit["condition"] for edits in target_groups.values() for edit in edits
    } == {
        "clean_original",
        "remux_only",
        "same_answer",
        "opposite_answer",
        "missing_source",
        "temporal_shift",
    }


def test_edit_plan_rejects_changed_anchor() -> None:
    pilot = _pilot()
    pilot["candidate_index"][0]["anchor"]["answer"] = "changed"
    pilot["attestation_sha256"] = _canonical_digest(
        {key: value for key, value in pilot.items() if key != "attestation_sha256"}
    )

    with pytest.raises(MediaEditValidationError, match="anchor"):
        _build_plan(pilot)


def test_edit_plan_accepts_permuted_donor_options_by_canonical_text() -> None:
    pilot = _pilot()
    pair = pilot["candidate_index"][0]
    donor = pair["donor"]
    target = dict(pair["target"])
    donor["question"] = f"  {donor['question'].upper()}  "
    donor["options"] = [" UNUSED ", "Beta", "ALPHA"]
    donor["answer"] = donor["answer"].upper()
    donor["answer_id"] = donor["options"].index(
        next(
            option
            for option in donor["options"]
            if option.strip().casefold() == donor["answer"].casefold()
        )
    )
    pilot["attestation_sha256"] = _canonical_digest(
        {key: value for key, value in pilot.items() if key != "attestation_sha256"}
    )

    plan = _build_plan(pilot)

    same_answer = next(
        edit
        for edit in plan
        if edit["target_id"] == f"{target['video_id']}:{target['question_id']}"
        and edit["condition"]
        == {
            "same_answer_nuisance": "same_answer",
            "opposite_answer_candidate": "opposite_answer",
        }[pair["role"]]
    )
    assert same_answer["anchor"] == {
        "question": target["question"],
        "options": target["options"],
        "answer_id": target["answer_id"],
        "answer": target["answer"],
    }
    assert same_answer["donor"]["options"] == ["UNUSED", "Beta", "ALPHA"]


def test_edit_plan_rejects_component_partition_crossing() -> None:
    pilot = _pilot()
    crossing_video = pilot["components"][0]["video_ids"][0]
    pilot["components"][1]["video_ids"].append(crossing_video)
    pilot["attestation_sha256"] = _canonical_digest(
        {key: value for key, value in pilot.items() if key != "attestation_sha256"}
    )

    with pytest.raises(MediaEditValidationError, match="component|partition"):
        _build_plan(pilot)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema_version", "schema version"),
        ("anchor_answer_id", "answer ID"),
    ],
)
def test_pilot_index_rejects_float_encoded_integers(
    mutation: str, message: str
) -> None:
    pilot = _pilot()
    if mutation == "schema_version":
        pilot["schema_version"] = 1.0
    else:
        pilot["candidate_index"][0]["anchor"]["answer_id"] = float(
            pilot["candidate_index"][0]["anchor"]["answer_id"]
        )
    pilot["attestation_sha256"] = _canonical_digest(
        {key: value for key, value in pilot.items() if key != "attestation_sha256"}
    )

    with pytest.raises(MediaEditValidationError, match=message):
        _build_plan(pilot)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("schema_version",), 4.0),
        (("video_encoding", "crf"), 18.0),
        (("container", "strip_metadata"), 1),
    ],
)
def test_configuration_rejects_noncanonical_scalar_types(
    field_path: tuple[str, ...], replacement: object
) -> None:
    configuration = _configuration()
    destination = configuration
    for key in field_path[:-1]:
        destination = destination[key]
    destination[field_path[-1]] = replacement

    with pytest.raises(MediaEditValidationError, match="integer|boolean"):
        media_edits._validate_configuration(configuration)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("media_file_count", 18.0, "media file count"),
        ("resumed_existing", 0, "resumed-existing"),
    ],
)
def test_media_receipt_rejects_noncanonical_scalar_types(
    tmp_path: pathlib.Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    receipt_path = pathlib.Path(inputs["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if field == "media_file_count":
        receipt[field] = replacement
    else:
        receipt["media_files"][0][field] = replacement
    validator = getattr(media_edits, "_validate_media_receipt_scalar_types", None)
    assert validator is not None, "media-receipt scalar validator is missing"

    with pytest.raises(MediaEditValidationError, match=message):
        validator(receipt)


@pytest.mark.parametrize(
    ("orientation", "padded_stream_selector"),
    [
        ("audio_over_video", "v"),
        ("video_over_audio", "a"),
    ],
)
def test_paired_edits_keep_longer_donor_and_pad_shorter_target(
    tmp_path: pathlib.Path,
    orientation: str,
    padded_stream_selector: str,
) -> None:
    edit = next(
        item
        for item in _build_plan(_pilot())
        if item["condition"] == "same_answer"
        and item["source_orientation"] == orientation
    )
    target_id = edit["target_video_id"]
    donor_id = edit["donor_video_id"]
    source_diagnostics = {
        target_id: {"duration_seconds": 5.0},
        donor_id: {"duration_seconds": 8.0},
    }
    duration_function = getattr(media_edits, "_output_duration_for_edit", None)
    assert duration_function is not None, "paired duration policy is missing"

    output_duration = duration_function(edit, source_diagnostics)
    arguments = _ffmpeg_arguments(
        ffmpeg_path=tmp_path / "ffmpeg",
        edit=edit,
        media_root=tmp_path,
        output_path=tmp_path / f"{orientation}.mp4",
        target_duration=5.0,
        target_stream_duration=5.0,
        donor_duration=8.0,
        output_duration=output_duration,
        target_diagnostic={"video": {"frame_rate": 25.0, "width": 640, "height": 480}},
        configuration=_configuration(),
    )
    command = " ".join(str(argument) for argument in arguments)

    assert output_duration == 8.0
    assert "-t 8.000000" in command
    assert "duration=5.000000" not in command
    assert f"-filter:{padded_stream_selector}" in arguments
    assert "apad" in command
    assert "tpad=" in command


def _write_fake_tools(
    root: pathlib.Path,
    *,
    fail_ffmpeg: bool = False,
    edited_sample_rate: int = 48000,
    edited_bitrate: int = 128000,
    edited_sample_format: str = "fltp",
    edited_container: str = "mov,mp4,m4a,3gp,3g2,mj2",
    edited_extra_stream: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    ffmpeg = root / "ffmpeg"
    if fail_ffmpeg:
        ffmpeg_body = (
            "#!/bin/sh\n"
            'if [ "$1" = "-version" ]; then '
            "echo 'ffmpeg version fixture'; exit 0; fi\n"
            "exit 7\n"
        )
    else:
        ffmpeg_body = """#!/bin/sh
if [ "$1" = "-version" ]; then
  echo 'ffmpeg version fixture'
  exit 0
fi
last=''
for value in "$@"; do last="$value"; done
printf 'deterministic-media:%s\n' "$(basename "$last")" > "$last"
"""
    ffmpeg.write_text(ffmpeg_body, encoding="utf-8")
    ffmpeg.chmod(0o755)
    ffprobe = root / "ffprobe"
    source_probe_payload = {
        "format": {
            "duration": "5.0",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "bit_rate": "256000",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 640,
                "height": 480,
                "avg_frame_rate": "25/1",
                "duration": "5.0",
                "bit_rate": "128000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_fmt": "fltp",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "5.0",
                "bit_rate": "128000",
            },
        ],
    }
    edited_probe_payload = json.loads(json.dumps(source_probe_payload))
    edited_probe_payload["streams"][1]["sample_rate"] = str(edited_sample_rate)
    edited_probe_payload["streams"][1]["bit_rate"] = str(edited_bitrate)
    edited_probe_payload["streams"][1]["sample_fmt"] = edited_sample_format
    edited_probe_payload["format"]["format_name"] = edited_container
    if edited_extra_stream:
        edited_probe_payload["streams"].append(
            {"codec_type": "subtitle", "codec_name": "mov_text"}
        )
    ffprobe.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-version" ]; then echo \'ffprobe version fixture\'; exit 0; fi\n'
        "last=''\n"
        'for value in "$@"; do last="$value"; done\n'
        'case "$last" in\n'
        "  */selected/*) printf '%s\\n' '"
        + json.dumps(source_probe_payload, separators=(",", ":"))
        + "' ;;\n"
        "  *) printf '%s\\n' '"
        + json.dumps(edited_probe_payload, separators=(",", ":"))
        + "' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    ffprobe.chmod(0o755)
    return ffmpeg, ffprobe


def _temporal_generator_request(source: dict, shift: float) -> dict:
    identity = {
        "component_id": source["component_id"],
        "partition": source["partition"],
        "video_id": source["video_id"],
        "source_role": source["source_role"],
        "shift_seconds": shift,
    }
    return {
        "schema": "conflictbench.perception-temporal-generator-request.v1",
        "event_id": _canonical_digest(identity)[:24],
        **identity,
        "source_filename": f"{source['video_id']}.mp4",
        "source_sha256": source["sha256"],
    }


def _temporal_scorer_request(
    source: dict,
    shift: float,
    *,
    generator_request_sha256: str,
    shifted_media_sha256: str,
) -> dict:
    identity = {
        "component_id": source["component_id"],
        "partition": source["partition"],
        "video_id": source["video_id"],
        "source_role": source["source_role"],
        "shift_seconds": shift,
    }
    return {
        "schema": "conflictbench.perception-temporal-scorer-request.v1",
        "event_id": _canonical_digest(identity)[:24],
        **identity,
        "generator_request_sha256": generator_request_sha256,
        "reference_media_sha256": source["sha256"],
        "shifted_media_sha256": shifted_media_sha256,
    }


def _write_temporal_fixture_sources(root: pathlib.Path) -> dict[str, pathlib.Path]:
    sources = {
        "generator": root / "temporal_generator.py",
        "scorer": root / "temporal_scorer.py",
    }
    sources["generator"].write_text(
        """#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True, type=pathlib.Path)
parser.add_argument("--source", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)
args = parser.parse_args()
request_bytes = args.request.read_bytes()
request = json.loads(request_bytes.decode("utf-8"))
source_bytes = args.source.read_bytes()
if hashlib.sha256(source_bytes).hexdigest() != request["source_sha256"]:
    raise SystemExit(2)
if args.source.name != request["source_filename"]:
    raise SystemExit(3)
shift = format(float(request["shift_seconds"]), ".17g").encode("ascii")
args.output.write_bytes(source_bytes + b"\\nshift_seconds=" + shift + b"\\n")
""",
        encoding="utf-8",
    )
    sources["scorer"].write_text(
        """#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True, type=pathlib.Path)
parser.add_argument("--reference", required=True, type=pathlib.Path)
parser.add_argument("--shifted", required=True, type=pathlib.Path)
args = parser.parse_args()
request_bytes = args.request.read_bytes()
request = json.loads(request_bytes.decode("utf-8"))
if hashlib.sha256(args.reference.read_bytes()).hexdigest() != request["reference_media_sha256"]:
    raise SystemExit(2)
if hashlib.sha256(args.shifted.read_bytes()).hexdigest() != request["shifted_media_sha256"]:
    raise SystemExit(3)
shifted_scores = {"0.25": 0.7, "0.5": 0.2, "1": 0.2}
response = {
    "schema": "conflictbench.perception-temporal-scorer-response.v1",
    "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
    "reference_score": 1.0,
    "shifted_score": shifted_scores[format(float(request["shift_seconds"]), ".17g")],
}
print(json.dumps(response, sort_keys=True, separators=(",", ":")))
""",
        encoding="utf-8",
    )
    return sources


def _write_inputs(tmp_path: pathlib.Path) -> dict[str, object]:
    pilot = _pilot()
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(pilot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(
        json.dumps(_configuration(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nuisance_path = tmp_path / "nuisance.json"
    nuisance_path.write_text(
        json.dumps(_nuisance_configuration(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    media_root = tmp_path / "selected"
    media_root.mkdir()
    files = []
    for record in _normalized_media(pilot):
        path = media_root / record["filename"]
        path.write_bytes(record["video_id"].encode("utf-8"))
        path.chmod(0o440)
        files.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _file_digest(path),
                "zip_crc32": "00000000",
                "resumed_existing": False,
                "stream_types": ["audio", "video"],
            }
        )
    normalized = [
        {
            "video_id": pathlib.Path(item["filename"]).stem,
            "filename": item["filename"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "stream_types": item["stream_types"],
        }
        for item in files
    ]
    media_set_sha256 = _canonical_digest(normalized)
    receipt = {
        "schema": "conflictbench.perception-pilot-media-receipt.v1",
        "status": "selected_media_extracted",
        "input_digests": {
            "train_archive_sha256": "f" * 64,
            "pilot_index_sha256": _file_digest(pilot_path),
        },
        "media_root_name": media_root.name,
        "media_file_count": len(files),
        "media_files": files,
        "stream_probe": {
            "filename": "ffprobe",
            "sha256": "e" * 64,
            "version": "ffprobe fixture",
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_gate_configuration_path = tmp_path / "source_gate_configuration.json"
    source_gate_configuration_path.write_text(
        json.dumps({"fixture": "source-gate-configuration"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_gate_source_root = tmp_path / "source_gate_sources"
    source_gate_source_root.mkdir()
    source_gate_sources = {
        "gate": source_gate_source_root / "perception_omni_gate.py",
        "runner": source_gate_source_root / "run_perception_omni_gate.py",
    }
    for role, source_path in source_gate_sources.items():
        source_path.write_text(f"# {role}\n", encoding="utf-8")
    source_gate_source_digests = {
        role: _file_digest(path) for role, path in source_gate_sources.items()
    }
    source_gate_source_bundle_sha256 = _canonical_digest(
        dict(sorted(source_gate_source_digests.items()))
    )
    source_gate_transcript_value = {"fixture": "source-gate-transcript"}
    source_gate_transcript_path = tmp_path / "source_gate_transcript.json"
    source_gate_transcript_path.write_text(
        json.dumps(source_gate_transcript_value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_gate_source_files = sorted(
        [
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": source_gate_source_digests[role],
            }
            for role, path in source_gate_sources.items()
        ],
        key=lambda item: item["filename"],
    )
    source_gate_output = {
        "gate": {
            "source_sufficiency_status": "pass",
            "nuisance_detection_status": "not_run",
            "overall_pilot_status": "pending_nuisance_detection",
            "metrics": {"audio_only": {}, "video_only": {}, "audiovisual": {}},
        },
        "input_digests": {
            "configuration_sha256": _file_digest(source_gate_configuration_path),
            "pilot_index_sha256": _file_digest(pilot_path),
            "media_set_sha256": media_set_sha256,
            "source_set_sha256": _canonical_digest(source_gate_source_files),
        },
        "source_files": source_gate_source_files,
        "media_files": normalized,
        "inference_transcript_sha256": _canonical_digest(source_gate_transcript_value),
    }
    source_gate_output_path = tmp_path / "source_gate_output.json"
    source_gate_output_path.write_text(
        json.dumps(source_gate_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    media_sha256_by_id = {item["video_id"]: item["sha256"] for item in normalized}
    scorer_fit_sources_by_id: dict[str, dict] = {}
    for pair in pilot["candidate_index"]:
        if pair["partition"] != "scorer_fit":
            continue
        for source_role in ("target", "donor"):
            source = pair[source_role]
            scorer_fit_sources_by_id[source["video_id"]] = {
                "component_id": pair["component_id"],
                "partition": "scorer_fit",
                "video_id": source["video_id"],
                "source_role": source_role,
                "sha256": media_sha256_by_id[source["video_id"]],
            }
    scorer_fit_sources = sorted(
        scorer_fit_sources_by_id.values(),
        key=lambda item: (item["component_id"], item["source_role"], item["video_id"]),
    )
    calibration_source_set_sha256 = _canonical_digest(scorer_fit_sources)
    temporal_source_root = tmp_path / "temporal_sources"
    temporal_source_root.mkdir()
    temporal_sources = _write_temporal_fixture_sources(temporal_source_root)
    temporal_source_digests = {
        role: _file_digest(path) for role, path in temporal_sources.items()
    }
    temporal_source_bundle_sha256 = _canonical_digest(
        dict(sorted(temporal_source_digests.items()))
    )
    candidate_shifts = [0.25, 0.5, 1.0]
    shifted_scores = {0.25: 0.7, 0.5: 0.2, 1.0: 0.2}
    temporal_events = []
    for shift in candidate_shifts:
        for source in scorer_fit_sources:
            identity = {
                "component_id": source["component_id"],
                "partition": source["partition"],
                "video_id": source["video_id"],
                "source_role": source["source_role"],
                "shift_seconds": shift,
            }
            generator_request = _temporal_generator_request(source, shift)
            source_bytes = pathlib.Path(
                media_root, generator_request["source_filename"]
            ).read_bytes()
            shifted_media = (
                source_bytes
                + b"\nshift_seconds="
                + format(shift, ".17g").encode("ascii")
                + b"\n"
            )
            generator_request_sha256 = _canonical_digest(generator_request)
            shifted_media_sha256 = hashlib.sha256(shifted_media).hexdigest()
            scorer_request = _temporal_scorer_request(
                source,
                shift,
                generator_request_sha256=generator_request_sha256,
                shifted_media_sha256=shifted_media_sha256,
            )
            scorer_request_sha256 = _canonical_digest(scorer_request)
            scorer_response = {
                "schema": "conflictbench.perception-temporal-scorer-response.v1",
                "request_sha256": scorer_request_sha256,
                "reference_score": 1.0,
                "shifted_score": shifted_scores[shift],
            }
            temporal_events.append(
                {
                    "event_id": _canonical_digest(identity)[:24],
                    **identity,
                    "source_sha256": source["sha256"],
                    "generator_request_sha256": generator_request_sha256,
                    "shifted_media_sha256": shifted_media_sha256,
                    "scorer_request_sha256": scorer_request_sha256,
                    "scorer_response_sha256": _canonical_digest(scorer_response),
                    "reference_score": 1.0,
                    "shifted_score": shifted_scores[shift],
                }
            )
    temporal_calibration = _attest(
        {
            "schema": "conflictbench.perception-temporal-shift-events.v2",
            "status": "complete",
            "official_split": "train",
            "calibration_partition": "scorer_fit",
            "replay_protocol": "pinned_python_json_cli_v1",
            "selection_rule": (
                "maximum_mean_absolute_score_change_then_smallest_shift_seconds"
            ),
            "event_scoring_rule": "absolute_reference_minus_shifted",
            "input_digests": {
                "pilot_index_sha256": _file_digest(pilot_path),
                "media_receipt_sha256": _file_digest(receipt_path),
                "media_set_sha256": media_set_sha256,
                "calibration_source_set_sha256": calibration_source_set_sha256,
                "source_gate_output_sha256": _file_digest(source_gate_output_path),
                "source_gate_transcript_sha256": _file_digest(
                    source_gate_transcript_path
                ),
                "temporal_implementation_bundle_sha256": (
                    temporal_source_bundle_sha256
                ),
            },
            "implementation_source_sha256": temporal_source_digests,
            "events": temporal_events,
            "candidate_scores": [
                {
                    "shift_seconds": 0.25,
                    "event_count": len(scorer_fit_sources),
                    "mean_absolute_score_change": 0.3,
                },
                {
                    "shift_seconds": 0.5,
                    "event_count": len(scorer_fit_sources),
                    "mean_absolute_score_change": 0.8,
                },
                {
                    "shift_seconds": 1.0,
                    "event_count": len(scorer_fit_sources),
                    "mean_absolute_score_change": 0.8,
                },
            ],
            "selected_shift_seconds": 0.5,
        }
    )
    temporal_calibration_path = tmp_path / "temporal_calibration.json"
    temporal_calibration_path.write_text(
        json.dumps(temporal_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    nuisance_source_root = tmp_path / "nuisance_sources"
    nuisance_source_root.mkdir()
    nuisance_sources: dict[str, pathlib.Path] = {}
    nuisance_source_digests: dict[str, str] = {}
    for index, role in enumerate(IMPLEMENTATION_SOURCE_ROLES):
        source_path = nuisance_source_root / f"source_{index}.py"
        source_path.write_text(f"# {role}\n", encoding="utf-8")
        nuisance_sources[role] = source_path
        nuisance_source_digests[role] = _file_digest(source_path)
    nuisance_bundle_sha256 = _canonical_digest(
        dict(sorted(nuisance_source_digests.items()))
    )
    nuisance_contract = build_nuisance_contract(
        configuration_sha256=_file_digest(nuisance_path),
        pilot_index_sha256=_file_digest(pilot_path),
        media_receipt_sha256=_file_digest(receipt_path),
        media_set_sha256=media_set_sha256,
        implementation_bundle_sha256=nuisance_bundle_sha256,
    )
    nuisance_contract_path = tmp_path / "nuisance_contract.json"
    nuisance_contract_path.write_text(
        json.dumps(nuisance_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "configuration_path": configuration_path,
        "configuration_sha256": _file_digest(configuration_path),
        "pilot_path": pilot_path,
        "pilot_sha256": _file_digest(pilot_path),
        "receipt_path": receipt_path,
        "receipt_sha256": _file_digest(receipt_path),
        "media_set_sha256": media_set_sha256,
        "media_root": media_root,
        "nuisance_path": nuisance_path,
        "nuisance_sha256": _file_digest(nuisance_path),
        "nuisance_sources": nuisance_sources,
        "nuisance_bundle_sha256": nuisance_bundle_sha256,
        "nuisance_contract_path": nuisance_contract_path,
        "nuisance_contract_sha256": _file_digest(nuisance_contract_path),
        "source_gate_configuration_path": source_gate_configuration_path,
        "source_gate_configuration_sha256": _file_digest(
            source_gate_configuration_path
        ),
        "source_gate_output_path": source_gate_output_path,
        "source_gate_output_sha256": _file_digest(source_gate_output_path),
        "source_gate_sources": source_gate_sources,
        "source_gate_source_digests": source_gate_source_digests,
        "source_gate_source_bundle_sha256": source_gate_source_bundle_sha256,
        "source_gate_transcript_path": source_gate_transcript_path,
        "source_gate_transcript_sha256": _file_digest(source_gate_transcript_path),
        "temporal_calibration_path": temporal_calibration_path,
        "temporal_calibration_sha256": _file_digest(temporal_calibration_path),
        "temporal_sources": temporal_sources,
        "temporal_source_digests": temporal_source_digests,
        "temporal_source_bundle_sha256": temporal_source_bundle_sha256,
    }


def _construct_kwargs(
    inputs: dict[str, object],
    ffmpeg: pathlib.Path,
    ffprobe: pathlib.Path,
    output_root: pathlib.Path,
) -> dict[str, object]:
    return {
        "configuration_path": inputs["configuration_path"],
        "expected_configuration_sha256": inputs["configuration_sha256"],
        "pilot_index_path": inputs["pilot_path"],
        "expected_pilot_index_sha256": inputs["pilot_sha256"],
        "media_receipt_path": inputs["receipt_path"],
        "expected_media_receipt_sha256": inputs["receipt_sha256"],
        "expected_media_set_sha256": inputs["media_set_sha256"],
        "media_root": inputs["media_root"],
        "nuisance_configuration_path": inputs["nuisance_path"],
        "expected_nuisance_configuration_sha256": inputs["nuisance_sha256"],
        "nuisance_implementation_sources": inputs["nuisance_sources"],
        "expected_nuisance_implementation_bundle_sha256": inputs[
            "nuisance_bundle_sha256"
        ],
        "nuisance_contract_path": inputs["nuisance_contract_path"],
        "expected_nuisance_contract_sha256": inputs["nuisance_contract_sha256"],
        "source_gate_configuration_path": inputs["source_gate_configuration_path"],
        "expected_source_gate_configuration_sha256": inputs[
            "source_gate_configuration_sha256"
        ],
        "source_gate_output_path": inputs["source_gate_output_path"],
        "expected_source_gate_output_sha256": inputs["source_gate_output_sha256"],
        "source_gate_implementation_sources": inputs["source_gate_sources"],
        "expected_source_gate_implementation_source_sha256": inputs[
            "source_gate_source_digests"
        ],
        "expected_source_gate_implementation_bundle_sha256": inputs[
            "source_gate_source_bundle_sha256"
        ],
        "source_gate_transcript_path": inputs["source_gate_transcript_path"],
        "expected_source_gate_transcript_sha256": inputs[
            "source_gate_transcript_sha256"
        ],
        "temporal_implementation_sources": inputs["temporal_sources"],
        "expected_temporal_implementation_source_sha256": inputs[
            "temporal_source_digests"
        ],
        "expected_temporal_implementation_bundle_sha256": inputs[
            "temporal_source_bundle_sha256"
        ],
        "temporal_calibration_path": inputs["temporal_calibration_path"],
        "expected_temporal_calibration_sha256": inputs["temporal_calibration_sha256"],
        "ffmpeg_path": ffmpeg,
        "ffprobe_path": ffprobe,
        "output_root": output_root,
    }


def _replace_nuisance_configuration(
    inputs: dict[str, object], configuration: dict
) -> None:
    nuisance_path = pathlib.Path(inputs["nuisance_path"])
    nuisance_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["nuisance_sha256"] = _file_digest(nuisance_path)
    contract = build_nuisance_contract(
        configuration_sha256=str(inputs["nuisance_sha256"]),
        pilot_index_sha256=str(inputs["pilot_sha256"]),
        media_receipt_sha256=str(inputs["receipt_sha256"]),
        media_set_sha256=str(inputs["media_set_sha256"]),
        implementation_bundle_sha256=str(inputs["nuisance_bundle_sha256"]),
    )
    contract_path = pathlib.Path(inputs["nuisance_contract_path"])
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["nuisance_contract_sha256"] = _file_digest(contract_path)


def _runner_arguments(
    inputs: dict[str, object],
    ffmpeg: pathlib.Path,
    ffprobe: pathlib.Path,
    output_root: pathlib.Path,
) -> list[str]:
    arguments = [
        sys.executable,
        str(RUNNER),
        "--configuration",
        str(inputs["configuration_path"]),
        "--configuration-sha256",
        str(inputs["configuration_sha256"]),
        "--pilot-index",
        str(inputs["pilot_path"]),
        "--pilot-index-sha256",
        str(inputs["pilot_sha256"]),
        "--media-receipt",
        str(inputs["receipt_path"]),
        "--media-receipt-sha256",
        str(inputs["receipt_sha256"]),
        "--media-set-sha256",
        str(inputs["media_set_sha256"]),
        "--media-root",
        str(inputs["media_root"]),
        "--nuisance-configuration",
        str(inputs["nuisance_path"]),
        "--nuisance-configuration-sha256",
        str(inputs["nuisance_sha256"]),
        "--nuisance-implementation-bundle-sha256",
        str(inputs["nuisance_bundle_sha256"]),
        "--nuisance-contract",
        str(inputs["nuisance_contract_path"]),
        "--nuisance-contract-sha256",
        str(inputs["nuisance_contract_sha256"]),
        "--source-gate-configuration",
        str(inputs["source_gate_configuration_path"]),
        "--source-gate-configuration-sha256",
        str(inputs["source_gate_configuration_sha256"]),
        "--source-gate-output",
        str(inputs["source_gate_output_path"]),
        "--source-gate-output-sha256",
        str(inputs["source_gate_output_sha256"]),
        "--source-gate-implementation-bundle-sha256",
        str(inputs["source_gate_source_bundle_sha256"]),
        "--source-gate-transcript",
        str(inputs["source_gate_transcript_path"]),
        "--source-gate-transcript-sha256",
        str(inputs["source_gate_transcript_sha256"]),
        "--temporal-implementation-bundle-sha256",
        str(inputs["temporal_source_bundle_sha256"]),
        "--temporal-calibration",
        str(inputs["temporal_calibration_path"]),
        "--temporal-calibration-sha256",
        str(inputs["temporal_calibration_sha256"]),
        "--ffmpeg",
        str(ffmpeg),
        "--ffprobe",
        str(ffprobe),
        "--output-root",
        str(output_root),
    ]
    for role, source_path in sorted(dict(inputs["nuisance_sources"]).items()):
        arguments.extend(["--nuisance-implementation-source", f"{role}={source_path}"])
    for role, source_path in sorted(dict(inputs["source_gate_sources"]).items()):
        arguments.extend(
            ["--source-gate-implementation-source", f"{role}={source_path}"]
        )
        arguments.extend(
            [
                "--source-gate-implementation-source-sha256",
                f"{role}={inputs['source_gate_source_digests'][role]}",
            ]
        )
    for role, source_path in sorted(dict(inputs["temporal_sources"]).items()):
        arguments.extend(["--temporal-implementation-source", f"{role}={source_path}"])
        arguments.extend(
            [
                "--temporal-implementation-source-sha256",
                f"{role}={inputs['temporal_source_digests'][role]}",
            ]
        )
    return arguments


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_constructs_and_revalidates_hash_bound_outputs(tmp_path: pathlib.Path) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    output_root = tmp_path / "controlled"
    kwargs = _construct_kwargs(inputs, ffmpeg, ffprobe, output_root)

    built = construct_controlled_media(**kwargs)
    verified = verify_controlled_media(**kwargs)

    assert built == verified
    assert built["schema"] == "conflictbench.perception-controlled-media-run.v4"
    assert built["status"] == "complete"
    assert built["continuation_allowed"] is True
    assert {
        "nuisance_configuration_sha256",
        "nuisance_implementation_bundle_sha256",
        "nuisance_contract_sha256",
        "source_gate_configuration_sha256",
        "source_gate_output_sha256",
        "source_gate_implementation_bundle_sha256",
        "source_gate_transcript_sha256",
        "temporal_implementation_bundle_sha256",
        "temporal_calibration_sha256",
    } <= set(built["input_digests"])
    assert built["nuisance_implementation_source_sha256"] == {
        role: _file_digest(path)
        for role, path in sorted(dict(inputs["nuisance_sources"]).items())
    }
    assert built["source_gate_implementation_source_sha256"] == dict(
        inputs["source_gate_source_digests"]
    )
    assert built["temporal_implementation_source_sha256"] == dict(
        inputs["temporal_source_digests"]
    )
    assert built["counts"] == {
        "edit_count": 60,
        "post_edit_pair_count": 24,
        "source_video_count": 18,
        "target_count": 6,
        "clean_original_count": 6,
    }
    assert len(list((output_root / "media").glob("*.mp4"))) == 60
    assert json.loads(
        (output_root / "post_edit_index.json").read_text(encoding="utf-8")
    )[0]["role"] in {
        "same_answer_nuisance",
        "opposite_answer_candidate",
    }
    validation = json.loads(
        (output_root / "validation.json").read_text(encoding="utf-8")
    )
    assert (
        validation["schema"]
        == "conflictbench.perception-controlled-media-validation.v4"
    )
    assert (
        validation["nuisance_reports"]["audio_over_video"]["diagnostic_stage"]
        == "final_edited_outputs"
    )
    assert (
        validation["nuisance_implementation_source_sha256"]
        == built["nuisance_implementation_source_sha256"]
    )
    edit_index = json.loads(
        (output_root / "edit_index.json").read_text(encoding="utf-8")
    )
    assert edit_index["schema"] == "conflictbench.perception-controlled-media-index.v4"
    clean = next(
        edit for edit in edit_index["edits"] if edit["condition"] == "clean_original"
    )
    source = pathlib.Path(inputs["media_root"]) / clean["source_files"][0]["filename"]
    assert (
        output_root / clean["output"]["relative_path"]
    ).read_bytes() == source.read_bytes()

    changed = next((output_root / "media").glob("*.mp4"))
    changed.chmod(0o600)
    changed.write_bytes(b"changed")
    with pytest.raises(MediaEditValidationError, match="digest|size|writable"):
        verify_controlled_media(**kwargs)


def _rewrite_json_file(path: pathlib.Path, value: object) -> str:
    path.chmod(0o600)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o400)
    return _file_digest(path)


def _reattest_record(value: dict) -> dict:
    value.pop("attestation_sha256", None)
    return _attest(value)


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_verify_rejects_unknown_edit_index_field_even_when_hashes_are_rebound(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    output_root = tmp_path / "controlled"
    kwargs = _construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
    construct_controlled_media(**kwargs)

    edit_index_path = output_root / "edit_index.json"
    edit_index = json.loads(edit_index_path.read_text(encoding="utf-8"))
    edit_index["unexpected"] = True
    edit_index_sha256 = _rewrite_json_file(
        edit_index_path, _reattest_record(edit_index)
    )

    validation_path = output_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["input_digests"]["edit_index_sha256"] = edit_index_sha256
    validation_sha256 = _rewrite_json_file(
        validation_path, _reattest_record(validation)
    )

    run_path = output_root / "run_record.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record["output_digests"]["edit_index_sha256"] = edit_index_sha256
    run_record["output_digests"]["validation_sha256"] = validation_sha256
    _rewrite_json_file(run_path, _reattest_record(run_record))

    with pytest.raises(MediaEditValidationError, match="edit index fields differ"):
        verify_controlled_media(**kwargs)


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_verify_rejects_float_counts_after_full_bundle_rebinding(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    output_root = tmp_path / "controlled"
    kwargs = _construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
    construct_controlled_media(**kwargs)

    edit_index_path = output_root / "edit_index.json"
    edit_index = json.loads(edit_index_path.read_text(encoding="utf-8"))
    edit_index["counts"]["edit_count"] = 60.0
    edit_index_sha256 = _rewrite_json_file(
        edit_index_path, _reattest_record(edit_index)
    )

    validation_path = output_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["input_digests"]["edit_index_sha256"] = edit_index_sha256
    validation["checks"]["edit_count"] = 60.0
    validation_sha256 = _rewrite_json_file(
        validation_path, _reattest_record(validation)
    )

    run_path = output_root / "run_record.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record["counts"]["edit_count"] = 60.0
    run_record["output_digests"]["edit_index_sha256"] = edit_index_sha256
    run_record["output_digests"]["validation_sha256"] = validation_sha256
    _rewrite_json_file(run_path, _reattest_record(run_record))

    with pytest.raises(MediaEditValidationError, match="positive integer"):
        verify_controlled_media(**kwargs)


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_generated_index_schema_rejects_unknown_nested_fields(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    output_root = tmp_path / "controlled"
    construct_controlled_media(
        **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
    )
    edit_index = json.loads(
        (output_root / "edit_index.json").read_text(encoding="utf-8")
    )
    post_edit = json.loads(
        (output_root / "post_edit_index.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (output_root / "validation.json").read_text(encoding="utf-8")
    )
    run_record = json.loads(
        (output_root / "run_record.json").read_text(encoding="utf-8")
    )
    mutations = []
    for location in ("entry", "source", "operation", "output"):
        mutated = copy.deepcopy(edit_index)
        if location == "entry":
            mutated["edits"][0]["unexpected"] = True
        elif location == "source":
            mutated["edits"][0]["source_files"][0]["unexpected"] = True
        elif location == "operation":
            mutated["edits"][0]["operation"]["unexpected"] = True
        else:
            mutated["edits"][0]["output"]["unexpected"] = True
        mutations.append((media_edits._validate_edit_index_schema, mutated))
    mutated_post = copy.deepcopy(post_edit)
    mutated_post[0]["unexpected"] = True
    mutations.append((media_edits._validate_post_edit_index_schema, mutated_post))
    mutated_validation = copy.deepcopy(validation)
    mutated_validation["checks"]["unexpected"] = True
    mutations.append((media_edits._validate_validation_schema, mutated_validation))
    mutated_run = copy.deepcopy(run_record)
    mutated_run["output_digests"]["unexpected"] = "0" * 64
    mutations.append((media_edits._validate_run_schema, mutated_run))

    for validator, mutated in mutations:
        if isinstance(mutated, dict) and "attestation_sha256" in mutated:
            mutated = _reattest_record(mutated)
        with pytest.raises(MediaEditValidationError, match="fields differ"):
            validator(mutated)


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_temporal_calibration_that_breaks_the_frozen_selection_rule(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    calibration_path = pathlib.Path(inputs["temporal_calibration_path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["selected_shift_seconds"] = 1.0
    calibration.pop("attestation_sha256")
    calibration = _attest(calibration)
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["temporal_calibration_sha256"] = _file_digest(calibration_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="deterministic rule"):
        construct_controlled_media(
            **_construct_kwargs(
                inputs, ffmpeg, ffprobe, tmp_path / "invalid-calibration"
            )
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_temporal_calibration_from_nonexecuting_pinned_sources(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    temporal_sources = dict(inputs["temporal_sources"])
    for role, source_path in temporal_sources.items():
        pathlib.Path(source_path).write_text(f"# {role}\n", encoding="utf-8")
    temporal_source_digests = {
        role: _file_digest(pathlib.Path(path))
        for role, path in temporal_sources.items()
    }
    temporal_bundle_sha256 = _canonical_digest(
        dict(sorted(temporal_source_digests.items()))
    )
    inputs["temporal_source_digests"] = temporal_source_digests
    inputs["temporal_source_bundle_sha256"] = temporal_bundle_sha256
    calibration_path = pathlib.Path(inputs["temporal_calibration_path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["implementation_source_sha256"] = temporal_source_digests
    calibration["input_digests"]["temporal_implementation_bundle_sha256"] = (
        temporal_bundle_sha256
    )
    _rewrite_temporal_calibration(inputs, calibration)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="temporal generator replay"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "output")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_temporal_event_hashes_that_differ_from_pinned_replay(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    calibration_path = pathlib.Path(inputs["temporal_calibration_path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["events"][0]["shifted_media_sha256"] = "0" * 64
    _rewrite_temporal_calibration(inputs, calibration)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="pinned replay"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "output")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_internally_consistent_self_reported_temporal_scores(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    calibration_path = pathlib.Path(inputs["temporal_calibration_path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    for event in calibration["events"]:
        event["shifted_score"] = 0.9
        response = {
            "schema": "conflictbench.perception-temporal-scorer-response.v1",
            "request_sha256": event["scorer_request_sha256"],
            "reference_score": event["reference_score"],
            "shifted_score": event["shifted_score"],
        }
        event["scorer_response_sha256"] = _canonical_digest(response)
    for candidate in calibration["candidate_scores"]:
        candidate["mean_absolute_score_change"] = 0.1
    calibration["selected_shift_seconds"] = 0.25
    _rewrite_temporal_calibration(inputs, calibration)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="pinned replay"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "output")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_numeric_strings_in_locked_json_schemas(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    configuration_path = pathlib.Path(inputs["configuration_path"])
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["validation"]["duration_tolerance_seconds"] = "0.1"
    inputs["configuration_sha256"] = _rewrite_json_file(
        configuration_path, configuration
    )

    with pytest.raises(MediaEditValidationError, match="finite number"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "output")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_integer_duration_tolerance(tmp_path: pathlib.Path) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    configuration_path = pathlib.Path(inputs["configuration_path"])
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["validation"]["duration_tolerance_seconds"] = 1
    inputs["configuration_sha256"] = _rewrite_json_file(
        configuration_path, configuration
    )

    with pytest.raises(MediaEditValidationError, match="finite float"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "output")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
@pytest.mark.parametrize(
    ("field_path", "replacement", "message"),
    [
        (("calibration_partition",), "pilot_gate", "contract"),
        (
            ("input_digests", "source_gate_output_sha256"),
            "0" * 64,
            "input binding",
        ),
    ],
)
def test_rejects_temporal_calibration_with_nontraining_or_unbound_provenance(
    tmp_path: pathlib.Path,
    field_path: tuple[str, ...],
    replacement: str,
    message: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    calibration_path = pathlib.Path(inputs["temporal_calibration_path"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    destination = calibration
    for key in field_path[:-1]:
        destination = destination[key]
    destination[field_path[-1]] = replacement
    calibration.pop("attestation_sha256")
    calibration = _attest(calibration)
    calibration_path.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["temporal_calibration_sha256"] = _file_digest(calibration_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match=message):
        construct_controlled_media(
            **_construct_kwargs(
                inputs, ffmpeg, ffprobe, tmp_path / "invalid-calibration"
            )
        )


def _rewrite_temporal_calibration(inputs: dict[str, object], value: dict) -> None:
    path = pathlib.Path(inputs["temporal_calibration_path"])
    value.pop("attestation_sha256", None)
    path.write_text(
        json.dumps(_attest(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inputs["temporal_calibration_sha256"] = _file_digest(path)


@pytest.mark.usefixtures("stub_source_gate_replay")
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "coverage"),
        ("duplicate", "duplicated"),
        ("wrong_partition", "scorer_fit"),
        ("wrong_source", "outside"),
        ("self_declared_score", "recomputation"),
    ],
)
def test_rejects_temporal_event_mutations(
    tmp_path: pathlib.Path, mutation: str, message: str
) -> None:
    inputs = _write_inputs(tmp_path)
    path = pathlib.Path(inputs["temporal_calibration_path"])
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        calibration["events"].pop()
    elif mutation == "duplicate":
        calibration["events"].append(dict(calibration["events"][0]))
    elif mutation == "wrong_partition":
        calibration["events"][0]["partition"] = "pilot_gate"
    elif mutation == "wrong_source":
        calibration["events"][0]["video_id"] = "unknown_video"
    else:
        calibration["candidate_scores"][0]["mean_absolute_score_change"] = 0.9
    _rewrite_temporal_calibration(inputs, calibration)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match=message):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / mutation)
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_tampered_temporal_scoring_source(tmp_path: pathlib.Path) -> None:
    inputs = _write_inputs(tmp_path)
    scorer = pathlib.Path(dict(inputs["temporal_sources"])["scorer"])
    scorer.write_text("# tampered scorer\n", encoding="utf-8")
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="source digest differs"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "tampered-scorer")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_changed_nuisance_implementation_source(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    source = next(iter(dict(inputs["nuisance_sources"]).values()))
    source.write_text("# changed\n", encoding="utf-8")
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="implementation-bundle"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "changed-source")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_changed_nuisance_preregistration_contract(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    contract_path = pathlib.Path(inputs["nuisance_contract_path"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "complete"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["nuisance_contract_sha256"] = _file_digest(contract_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="preregistered state"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "changed-contract")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_source_orientations_are_derived_from_replayed_gate(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    output_root = tmp_path / "derived-orientations"

    result = construct_controlled_media(
        **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
    )

    assert result["counts"]["edit_count"] == 60
    edit_index = json.loads(
        (output_root / "edit_index.json").read_text(encoding="utf-8")
    )
    assert {
        edit["source_orientation"]
        for edit in edit_index["edits"]
        if edit["source_orientation"] is not None
    } == {"audio_over_video", "video_over_audio"}


def test_source_gate_consumer_uses_real_authenticated_replay(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    support_path = pathlib.Path(__file__).with_name("test_perception_omni_gate.py")
    spec = importlib.util.spec_from_file_location(
        "omni_gate_test_support", support_path
    )
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    monkeypatch.setattr(
        omni_gate,
        "_probe_media_streams",
        lambda _path: ["audio", "video"],
    )
    config_path, config_sha256, pilot_path, pilot_sha256, media_root = (
        support._write_gate_inputs(tmp_path / "source-gate")
    )
    source_paths = {
        "gate": pathlib.Path(omni_gate.__file__),
        "runner": pathlib.Path(__file__).parents[1]
        / "scripts"
        / "run_perception_omni_gate.py",
    }
    output = omni_gate.run_source_sufficiency_gate(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        pilot_index_path=pilot_path,
        expected_pilot_index_sha256=pilot_sha256,
        media_root=media_root,
        backend=support._FakeBackend(),
        source_paths=[source_paths[role] for role in sorted(source_paths)],
    )
    output_path = tmp_path / "source_gate_output.json"
    output_path.touch()
    output_sha256 = _rewrite_json_file(output_path, output)
    transcript_path = tmp_path / "source_gate_transcript.json"
    transcript_path.touch()
    transcript_sha256 = _rewrite_json_file(
        transcript_path, omni_gate.build_inference_transcript(output)
    )
    source_digests = {role: _file_digest(path) for role, path in source_paths.items()}

    result = media_edits._validate_source_gate_evidence(
        configuration=_configuration(),
        source_gate_output_path=output_path,
        expected_source_gate_output_sha256=output_sha256,
        source_gate_configuration_path=config_path,
        expected_source_gate_configuration_sha256=config_sha256,
        pilot_index_path=pilot_path,
        expected_pilot_index_sha256=pilot_sha256,
        media_root=media_root,
        expected_media_set_sha256=output["input_digests"]["media_set_sha256"],
        media_records=output["media_files"],
        source_gate_implementation_sources=source_paths,
        expected_source_gate_implementation_source_sha256=source_digests,
        expected_source_gate_implementation_bundle_sha256=_canonical_digest(
            dict(sorted(source_digests.items()))
        ),
        source_gate_transcript_path=transcript_path,
        expected_source_gate_transcript_sha256=transcript_sha256,
    )

    assert result["allowed_orientations"] == [
        "audio_over_video",
        "video_over_audio",
    ]


def test_rejects_self_attested_source_gate_without_replay_pass(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    def reject_replay(_output: dict, **_kwargs: object) -> dict:
        raise media_edits.GateValidationError("replay differs")

    monkeypatch.setattr(media_edits, "verify_gate_output_replay", reject_replay)

    with pytest.raises(MediaEditValidationError, match="replay verification failed"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "rejected")
        )


@pytest.mark.parametrize(
    "input_name",
    [
        "source_gate_configuration_path",
        "source_gate_output_path",
        "source_gate_transcript_path",
    ],
)
def test_rejects_tampered_source_gate_files_even_with_valid_internal_json(
    tmp_path: pathlib.Path,
    input_name: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    path = pathlib.Path(inputs[input_name])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["unexpected"] = True
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="digest differs"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "tampered-gate")
        )


def test_rejects_tampered_source_gate_implementation_file(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    source = next(iter(dict(inputs["source_gate_sources"]).values()))
    source.write_text("# tampered\n", encoding="utf-8")
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="source digest differs"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "tampered-source")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_unregistered_source_gate_roles(tmp_path: pathlib.Path) -> None:
    inputs = _write_inputs(tmp_path)
    original = dict(inputs["source_gate_sources"])
    rebound = {"alpha": original["gate"], "beta": original["runner"]}
    rebound_digests = {role: _file_digest(path) for role, path in rebound.items()}
    inputs["source_gate_sources"] = rebound
    inputs["source_gate_source_digests"] = rebound_digests
    inputs["source_gate_source_bundle_sha256"] = _canonical_digest(
        dict(sorted(rebound_digests.items()))
    )
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="source roles"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "wrong-roles")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_source_media_changed_after_initial_authentication(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    media_root = pathlib.Path(inputs["media_root"])
    source_path = min(media_root.glob("*.mp4"))
    real_probe_sources = media_edits._probe_sources

    def mutate_after_probe(**kwargs):
        diagnostics = real_probe_sources(**kwargs)
        source_path.chmod(0o600)
        source_path.write_bytes(b"changed-after-authentication")
        source_path.chmod(0o400)
        return diagnostics

    monkeypatch.setattr(media_edits, "_probe_sources", mutate_after_probe)
    output_root = tmp_path / "changed-source"

    with pytest.raises(MediaEditValidationError, match="source media.*differs"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
        )
    assert not output_root.exists()


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_source_gate_failure_instead_of_accepting_orientation_assertion(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    output_path = pathlib.Path(inputs["source_gate_output_path"])
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["gate"]["source_sufficiency_status"] = "fail"
    output["gate"]["overall_pilot_status"] = "fail"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["source_gate_output_sha256"] = _file_digest(output_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match="does not authorize"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / "failed-gate")
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("configuration", "configuration_sha256 binding"),
        ("pilot", "pilot_index_sha256 binding"),
        ("media_set", "media_set_sha256 binding"),
        ("source_set", "source-set binding"),
        ("source_file", "implementation binding"),
        ("media_file", "media receipt binding"),
        ("transcript", "transcript binding"),
    ],
)
def test_rejects_replayed_source_gate_with_rebound_but_wrong_input_identity(
    tmp_path: pathlib.Path, mutation: str, message: str
) -> None:
    inputs = _write_inputs(tmp_path)
    output_path = pathlib.Path(inputs["source_gate_output_path"])
    output = json.loads(output_path.read_text(encoding="utf-8"))
    if mutation == "configuration":
        output["input_digests"]["configuration_sha256"] = "0" * 64
    elif mutation == "pilot":
        output["input_digests"]["pilot_index_sha256"] = "0" * 64
    elif mutation == "media_set":
        output["input_digests"]["media_set_sha256"] = "0" * 64
    elif mutation == "source_set":
        output["input_digests"]["source_set_sha256"] = "0" * 64
    elif mutation == "source_file":
        output["source_files"][0]["sha256"] = "0" * 64
    elif mutation == "media_file":
        output["media_files"][0]["sha256"] = "0" * 64
    else:
        output["inference_transcript_sha256"] = "0" * 64
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs["source_gate_output_sha256"] = _file_digest(output_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)

    with pytest.raises(MediaEditValidationError, match=message):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, tmp_path / mutation)
        )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_negative_nuisance_result_is_published_and_blocks_continuation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    nuisance = _nuisance_configuration()
    nuisance["gate"] = {
        "maximum_primary_upper_confidence_bound": 0.0,
        "maximum_unseen_question_upper_confidence_bound": 0.0,
        "maximum_point_lift_over_structural_chance": 0.0,
        "maximum_structural_chance_balanced_accuracy": 0.0,
        "minimum_permutation_p_value_exclusive": 1.0,
    }
    _replace_nuisance_configuration(inputs, nuisance)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    output_root = tmp_path / "negative"
    kwargs = _construct_kwargs(inputs, ffmpeg, ffprobe, output_root)

    built = construct_controlled_media(**kwargs)
    verified = verify_controlled_media(**kwargs)

    assert output_root.is_dir()
    assert built == verified
    assert built["status"] == "complete"
    assert built["continuation_allowed"] is False
    assert (
        json.loads((output_root / "validation.json").read_text(encoding="utf-8"))[
            "continuation_allowed"
        ]
        is False
    )
    spec = importlib.util.spec_from_file_location("controlled_media_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    monkeypatch.setattr(
        runner_module,
        "verify_controlled_media",
        lambda **_kwargs: {"continuation_allowed": False},
    )
    assert (
        runner_module.main(
            [
                *_runner_arguments(inputs, ffmpeg, ffprobe, output_root)[2:],
                "--verify-existing",
            ]
        )
        == 3
    )


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_failed_ffmpeg_does_not_publish_output_root(tmp_path: pathlib.Path) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path, fail_ffmpeg=True)
    output_root = tmp_path / "controlled"

    with pytest.raises(MediaEditValidationError, match="ffmpeg"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    "relative_output", [".", "controlled", "nested/controlled", ".."]
)
def test_rejects_output_root_equal_to_below_or_above_media_root_before_loading_inputs(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_output: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path)
    media_root = pathlib.Path(inputs["media_root"])
    output_root = media_root / relative_output

    def fail_if_loaded(**_kwargs: object) -> dict:
        raise AssertionError("input loading must not run for an unsafe output root")

    monkeypatch.setattr(media_edits, "_load_context", fail_if_loaded)

    with pytest.raises(MediaEditValidationError, match="media root"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
        )
    assert not (media_root / "controlled").exists()


@pytest.mark.usefixtures("stub_source_gate_replay")
def test_rejects_edited_audio_that_differs_from_the_locked_encoding(
    tmp_path: pathlib.Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path, edited_sample_rate=44100)
    output_root = tmp_path / "controlled"

    with pytest.raises(MediaEditValidationError, match="sample rate"):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
        )
    assert not output_root.exists()


@pytest.mark.usefixtures("stub_source_gate_replay")
@pytest.mark.parametrize(
    ("tool_options", "message"),
    [
        ({"edited_bitrate": 32000}, "bitrate"),
        ({"edited_sample_format": "s16"}, "sample format"),
        ({"edited_container": "matroska,webm"}, "container"),
        ({"edited_extra_stream": True}, "stream"),
    ],
)
def test_rejects_outputs_outside_the_locked_media_profile(
    tmp_path: pathlib.Path, tool_options: dict, message: str
) -> None:
    inputs = _write_inputs(tmp_path)
    ffmpeg, ffprobe = _write_fake_tools(tmp_path, **tool_options)
    output_root = tmp_path / "controlled"

    with pytest.raises(MediaEditValidationError, match=message):
        construct_controlled_media(
            **_construct_kwargs(inputs, ffmpeg, ffprobe, output_root)
        )
    assert not output_root.exists()


def test_runner_help_exposes_explicit_tool_and_hash_arguments() -> None:
    assert os.access(RUNNER, os.X_OK)
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        env={
            **os.environ,
            "PYTHONPATH": str(pathlib.Path(__file__).parents[1] / "src"),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    for option in (
        "--ffmpeg",
        "--ffprobe",
        "--configuration-sha256",
        "--pilot-index-sha256",
        "--media-receipt-sha256",
        "--media-set-sha256",
        "--nuisance-configuration-sha256",
        "--nuisance-implementation-source",
        "--nuisance-implementation-bundle-sha256",
        "--nuisance-contract-sha256",
        "--source-gate-configuration-sha256",
        "--source-gate-output-sha256",
        "--source-gate-implementation-source",
        "--source-gate-implementation-source-sha256",
        "--source-gate-implementation-bundle-sha256",
        "--source-gate-transcript-sha256",
        "--temporal-implementation-source",
        "--temporal-implementation-source-sha256",
        "--temporal-implementation-bundle-sha256",
        "--temporal-calibration-sha256",
        "--verify-existing",
    ):
        assert option in completed.stdout


def test_real_ffmpeg_build_is_profile_valid_and_byte_reproducible(
    tmp_path: pathlib.Path,
) -> None:
    ffmpeg_value = shutil.which("ffmpeg")
    ffprobe_value = shutil.which("ffprobe")
    if ffmpeg_value is None or ffprobe_value is None:
        pytest.skip("ffmpeg and ffprobe are not both available")
    ffmpeg = pathlib.Path(ffmpeg_value).resolve()
    ffprobe = pathlib.Path(ffprobe_value).resolve()
    source = tmp_path / "source.mp4"
    generated = subprocess.run(
        [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            "-fflags",
            "+bitexact",
            "-threads",
            "1",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert generated.returncode == 0, generated.stderr
    source_diagnostic = media_edits.probe_media_file(
        ffprobe, source, expected_size_bytes=source.stat().st_size
    )
    edit = {
        "condition": "temporal_shift",
        "source_orientation": "audio_over_video",
        "operation": {
            "audio_source": "target",
            "video_source": "target",
            "audio_delay_seconds": 0.05,
        },
        "source_files": [{"filename": source.name}],
    }
    outputs = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    for output in outputs:
        completed = subprocess.run(
            _ffmpeg_arguments(
                ffmpeg_path=ffmpeg,
                edit=edit,
                media_root=tmp_path,
                output_path=output,
                target_duration=float(source_diagnostic["duration_seconds"]),
                target_stream_duration=float(source_diagnostic["duration_seconds"]),
                donor_duration=None,
                output_duration=float(source_diagnostic["duration_seconds"]),
                target_diagnostic=source_diagnostic,
                configuration=_configuration(),
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        profile = _probe_media_profile(ffprobe, output, _configuration())
        assert profile["audio"]["codec"] == "aac"
        assert profile["audio"]["sample_rate"] == 48000
        assert profile["audio"]["channels"] == 2
        assert profile["audio"]["sample_format"] == "fltp"
        assert profile["container_format_names"] == [
            "mov",
            "mp4",
            "m4a",
            "3gp",
            "3g2",
            "mj2",
        ]
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
