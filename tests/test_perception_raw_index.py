from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import typing
from pathlib import Path

import pytest

import conflictbench.perception_raw_index as raw_index
from conflictbench.perception_raw_index import (
    ACQUISITION_EXAMPLE_SCHEMA,
    CONFLICT_EXAMPLE_SCHEMA,
    POLICY_FEATURE_ALLOWLIST,
    POLICY_FEATURE_PAYLOAD_SCHEMA,
    POLICY_SAFE_VIEW_SCHEMA,
    ObservedMedia,
    ObservedMediaRequest,
    ObservedPolicyFeatures,
    PairedImpossibilityInput,
    PairedImpossibilityRecord,
    PolicyObservedMedia,
    PolicySafeView,
    RawIndexValidationError,
    build_acquisition_example,
    build_conflict_example,
    build_observed_policy_features,
    build_paired_impossibility_input,
    build_policy_safe_view,
    canonical_question_key_sha256,
    validate_acquisition_collection,
    validate_acquisition_example,
    validate_conflict_collection,
    validate_conflict_example,
    validate_paired_impossibility_collection,
    validate_raw_index_collections,
)


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


OBSERVED_BYTES = b"observed"
OBSERVED_MEDIA_SHA256 = _bytes_digest(OBSERVED_BYTES)


def _candidate_media_sha256(video_id: str) -> str:
    return _bytes_digest(f"candidate:{video_id}".encode())


def _question_key(text: str, options: list[str]) -> str:
    normalized_text = " ".join(text.split())
    normalized_options = sorted(" ".join(option.split()) for option in options)
    encoded = json.dumps(
        {"options": normalized_options, "question": normalized_text},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_payload(
    pre_query_output_sha256: str = "b" * 64,
    **feature_overrides,
) -> bytes:
    features = {
        "choice_probability_a": 0.10,
        "choice_probability_b": 0.70,
        "choice_probability_c": 0.20,
        "entropy": 0.8018185525433373,
        "top_two_margin": 0.50,
        "duration_seconds": 3.25,
        "question_token_count": 3,
        "option_token_count_mean": 1.0,
    }
    features.update(feature_overrides)
    return json.dumps(
        {
            "features": features,
            "pre_query_output_sha256": pre_query_output_sha256,
            "schema": POLICY_FEATURE_PAYLOAD_SCHEMA,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _acquisition_example() -> dict:
    question = {
        "text": "  Which\t event happened?  ",
        "options": [" first ", "second", "third"],
        "option_permutation": [2, 0, 1],
        "gold_answer": "B",
    }
    return build_acquisition_example(
        example_id="acquisition-001",
        partitions={"official": "train", "analysis": "router_fit"},
        bindings={
            "component_id": "component-001",
            "question_key_sha256": _question_key(question["text"], question["options"]),
            "observed_media_sha256": OBSERVED_MEDIA_SHA256,
            "candidate_video_id": "donor-same",
            "candidate_media_sha256": _candidate_media_sha256("donor-same"),
        },
        target={"video_id": "video-001", "question_id": 2},
        question=question,
        modalities={"observed": "audio", "candidate": "video"},
        observed_condition="clean_original",
        outputs={
            "pre_query": {
                "output_id": "pre-001",
                "sha256": "b" * 64,
                "feature_payload_sha256": hashlib.sha256(
                    _feature_payload()
                ).hexdigest(),
            },
            "post_query": {"output_id": "post-001", "sha256": "c" * 64},
        },
        acquisition_cost=0.25,
    )


def _conflict_example() -> dict:
    question = {
        "text": "  Which event happened? ",
        "options": ["first", " second ", "third"],
        "option_permutation": [0, 1, 2],
    }
    return build_conflict_example(
        example_id="conflict-001",
        partitions={"official": "validation", "analysis": "evaluation"},
        bindings={
            "target_video_id": "video-target",
            "target_question_id": 4,
            "pair_id": "pair-001",
            "component_id": "component-001",
            "question_key_sha256": _question_key(question["text"], question["options"]),
            "source_media_sha256": {
                "audio": _candidate_media_sha256("video-donor"),
                "video": OBSERVED_MEDIA_SHA256,
            },
        },
        condition="opposite_answer",
        orientation="audio_over_video",
        question=question,
        physical_sources={
            "audio": {"role": "donor", "video_id": "video-donor"},
            "video": {"role": "target", "video_id": "video-target"},
        },
        source_answers={"audio": "A", "video": "B"},
        integrated={"relation": "CONFLICT", "action": "ABSTAIN"},
        output={"sha256": "e" * 64},
    )


class _MediaAccessSpy:
    def __init__(self) -> None:
        self.observed_requests: list[object] = []
        self.candidate_calls = 0

    def load_observed(self, request: object) -> ObservedMedia:
        self.observed_requests.append(request)
        return ObservedMedia(
            modality=request.modality,
            observed_bytes=OBSERVED_BYTES,
            sha256=OBSERVED_MEDIA_SHA256,
        )

    def load_candidate(self, request: object) -> str:
        self.candidate_calls += 1
        raise AssertionError(f"candidate access before QUERY: {request!r}")


def _observed_features() -> ObservedPolicyFeatures:
    payload = _feature_payload()
    return _observed_features_from_payload(payload)


def _observed_features_from_payload(payload: bytes) -> ObservedPolicyFeatures:
    return build_observed_policy_features(
        payload,
        expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _inject_unknown(value: dict, path: tuple[str, ...]) -> dict:
    changed = copy.deepcopy(value)
    current = changed
    for key in path:
        current = current[key]
    current["unexpected"] = True
    return changed


def test_acquisition_builder_canonicalizes_text_and_validates_the_versioned_record():
    example = _acquisition_example()

    assert example["schema"] == ACQUISITION_EXAMPLE_SCHEMA
    assert example["question"]["text"] == "Which event happened?"
    assert example["question"]["options"] == ["first", "second", "third"]
    assert validate_acquisition_example(example) == example


def test_question_key_hashes_canonical_question_and_permutation_invariant_options():
    expected = _question_key("Which event happened?", ["first", "second", "third"])

    assert (
        canonical_question_key_sha256(
            "  Which\t event happened?  ", [" first ", "second", "third"]
        )
        == expected
    )
    assert (
        canonical_question_key_sha256(
            "Which event happened?", ["second", "first", "third"]
        )
        == expected
    )


@pytest.mark.parametrize(
    "options",
    [None, "first second third", {"A": "first"}, ("first", "second")],
)
def test_question_key_rejects_noncanonical_option_collections(options):
    with pytest.raises(RawIndexValidationError, match="options"):
        canonical_question_key_sha256("Which event happened?", options)


@pytest.mark.parametrize(
    ("official", "analysis"),
    [
        ("train", "semantic_fit"),
        ("train", "router_fit"),
        ("train", "threshold_calibration"),
        ("validation", "evaluation"),
    ],
)
def test_final_partition_matrix_accepts_only_legal_pairs(official, analysis):
    example = _acquisition_example()
    example["partitions"] = {"official": official, "analysis": analysis}

    assert validate_acquisition_example(example) == example


@pytest.mark.parametrize(
    ("official", "analysis"),
    [
        ("train", "evaluation"),
        ("train", "scorer_fit"),
        ("train", "pilot_gate"),
        ("validation", "semantic_fit"),
        ("validation", "router_fit"),
        ("test", "evaluation"),
    ],
)
def test_final_partition_matrix_rejects_illegal_pairs(official, analysis):
    example = _acquisition_example()
    example["partitions"] = {"official": official, "analysis": analysis}

    with pytest.raises(RawIndexValidationError, match="partition pair"):
        validate_acquisition_example(example)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("official", ["train"]), ("analysis", {"role": "router_fit"})],
)
def test_partition_matrix_rejects_nonscalar_roles(field, replacement):
    example = _acquisition_example()
    example["partitions"][field] = replacement

    with pytest.raises(RawIndexValidationError, match="partition"):
        validate_acquisition_example(example)


def test_question_key_binding_rejects_semantic_changes_but_accepts_option_reordering():
    changed_text = _acquisition_example()
    changed_text["question"]["text"] = "What happened next?"
    with pytest.raises(RawIndexValidationError, match="question_key_sha256"):
        validate_acquisition_example(changed_text)

    changed_option = _acquisition_example()
    changed_option["question"]["options"] = ["second", "first", "different"]
    with pytest.raises(RawIndexValidationError, match="question_key_sha256"):
        validate_acquisition_example(changed_option)

    reordered = _acquisition_example()
    reordered["question"]["options"] = ["second", "first", "third"]
    reordered["question"]["option_permutation"] = [0, 2, 1]
    assert validate_acquisition_example(reordered) == reordered


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("partitions",),
        ("bindings",),
        ("target",),
        ("question",),
        ("modalities",),
        ("outputs",),
        ("outputs", "pre_query"),
        ("outputs", "post_query"),
    ],
)
def test_acquisition_validator_rejects_unknown_keys_at_every_object_level(path):
    with pytest.raises(RawIndexValidationError, match="keys differ"):
        validate_acquisition_example(_inject_unknown(_acquisition_example(), path))


def test_acquisition_validator_rejects_noncomplementary_modalities_and_bad_permutation():
    same_modality = _acquisition_example()
    same_modality["modalities"]["candidate"] = "audio"
    with pytest.raises(RawIndexValidationError, match="must differ"):
        validate_acquisition_example(same_modality)

    repeated_option = _acquisition_example()
    repeated_option["question"]["option_permutation"] = [0, 0, 2]
    with pytest.raises(RawIndexValidationError, match="permutation"):
        validate_acquisition_example(repeated_option)


def test_acquisition_cost_rejects_integer_and_boolean_encodings():
    for value in (0, 1, False, True):
        example = _acquisition_example()
        example["acquisition_cost"] = value

        with pytest.raises(RawIndexValidationError, match="finite float"):
            validate_acquisition_example(example)


def test_record_scalar_fields_reject_stateful_string_subclasses():
    class _LeakyString(str):
        pass

    leaky_modality = _LeakyString("audio")
    leaky_modality.candidate_bytes = b"forbidden"
    example = _acquisition_example()
    example["modalities"]["observed"] = leaky_modality

    with pytest.raises(RawIndexValidationError, match="modalities"):
        validate_acquisition_example(example)


@pytest.mark.parametrize(
    "unsafe_identifier",
    [
        "/private/video.wav",
        "../secret",
        "~/hidden",
        "file:///tmp/video.wav",
        "https://example.test/video",
        "s3://bucket/key",
        "video/subdir",
        r"C:\\private\\video.wav",
        "private-video.wav.mp4",
    ],
)
def test_record_identifiers_reject_paths_and_uris(unsafe_identifier):
    for path in (
        ("example_id",),
        ("bindings", "component_id"),
        ("target", "video_id"),
        ("outputs", "pre_query", "output_id"),
    ):
        example = _acquisition_example()
        current = example
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = unsafe_identifier

        with pytest.raises(RawIndexValidationError, match="identifier"):
            validate_acquisition_example(example)


def test_policy_safe_view_has_only_allowed_fields_and_never_loads_candidate_media():
    access = _MediaAccessSpy()

    view = build_policy_safe_view(
        _acquisition_example(),
        observed_features=_observed_features(),
        media_access=access,
    )

    assert {field.name for field in dataclasses.fields(view)} == {
        "schema",
        "question",
        "options",
        "observed_input",
        "observed_modality",
        "candidate_modality",
        "acquisition_cost",
        "choice_probability_a",
        "choice_probability_b",
        "choice_probability_c",
        "entropy",
        "top_two_margin",
        "duration_seconds",
        "question_token_count",
        "option_token_count_mean",
    }
    assert view.schema == POLICY_SAFE_VIEW_SCHEMA
    assert view.observed_input == PolicyObservedMedia(
        modality="audio",
        observed_bytes=OBSERVED_BYTES,
    )
    assert view.question == "Which event happened?"
    assert view.options == ("first", "second", "third")
    assert access.candidate_calls == 0
    assert len(access.observed_requests) == 1
    request = access.observed_requests[0]
    assert request.video_id == "video-001"
    assert request.question_id == 2
    assert request.modality == "audio"
    assert request.condition == "clean_original"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("example_id",), "changed-example"),
        (("partitions", "analysis"), "semantic_fit"),
        (("bindings", "component_id"), "changed-component"),
        (("target", "video_id"), "changed-video"),
        (("target", "question_id"), 9),
        (("question", "option_permutation"), [1, 2, 0]),
        (("question", "gold_answer"), "A"),
        (("observed_condition",), "temporal_shift"),
        (("outputs", "pre_query", "output_id"), "changed-pre"),
        (("outputs", "post_query", "output_id"), "changed-post"),
        (("outputs", "post_query", "sha256"), "2" * 64),
    ],
)
def test_forbidden_record_field_mutations_cannot_change_policy_features(
    path, replacement
):
    original = _acquisition_example()
    changed = copy.deepcopy(original)
    current = changed
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement

    original_view = build_policy_safe_view(
        original,
        observed_features=_observed_features(),
        media_access=_MediaAccessSpy(),
    )
    changed_view = build_policy_safe_view(
        changed,
        observed_features=_observed_features(),
        media_access=_MediaAccessSpy(),
    )

    assert changed_view == original_view


def test_policy_feature_contract_is_an_exact_frozen_typed_allowlist():
    assert POLICY_FEATURE_ALLOWLIST == (
        "choice_probability_a",
        "choice_probability_b",
        "choice_probability_c",
        "entropy",
        "top_two_margin",
        "duration_seconds",
        "question_token_count",
        "option_token_count_mean",
    )
    assert dataclasses.is_dataclass(ObservedPolicyFeatures)
    assert ObservedPolicyFeatures.__dataclass_params__.frozen is True

    forbidden_mapping = dataclasses.asdict(_observed_features())
    forbidden_mapping["candidate_choice_probabilities"] = [0.1, 0.2, 0.7]
    with pytest.raises(RawIndexValidationError, match="ObservedPolicyFeatures"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=forbidden_mapping,
            media_access=_MediaAccessSpy(),
        )


def test_feature_payload_builder_parses_canonical_json_and_authenticates_its_bytes():
    payload = _feature_payload()
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    features = build_observed_policy_features(
        payload,
        expected_payload_sha256=payload_sha256,
    )

    assert features.payload_bytes == payload
    assert features.payload_sha256 == payload_sha256
    assert features.pre_query_output_sha256 == "b" * 64
    assert features.choice_probability_b == 0.70


def test_feature_payload_builder_rejects_noncanonical_unbound_or_ambiguous_json():
    canonical = _feature_payload()
    decoded = json.loads(canonical)
    noncanonical = json.dumps(decoded, sort_keys=False, indent=2).encode("utf-8")
    unknown = copy.deepcopy(decoded)
    unknown["features"]["candidate_probability"] = 0.9
    unknown_bytes = json.dumps(unknown, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    duplicate = canonical[:-1] + b',"schema":"duplicate"}'

    with pytest.raises(RawIndexValidationError, match="digest"):
        build_observed_policy_features(
            canonical,
            expected_payload_sha256="f" * 64,
        )
    with pytest.raises(RawIndexValidationError, match="canonical"):
        build_observed_policy_features(
            noncanonical,
            expected_payload_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )
    with pytest.raises(RawIndexValidationError, match="keys differ"):
        build_observed_policy_features(
            unknown_bytes,
            expected_payload_sha256=hashlib.sha256(unknown_bytes).hexdigest(),
        )
    with pytest.raises(RawIndexValidationError, match="duplicate"):
        build_observed_policy_features(
            duplicate,
            expected_payload_sha256=hashlib.sha256(duplicate).hexdigest(),
        )


def test_feature_payload_state_cannot_be_changed_after_authentication():
    features = _observed_features()
    object.__setattr__(features, "choice_probability_a", 0.20)

    with pytest.raises(RawIndexValidationError, match="payload"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=features,
            media_access=_MediaAccessSpy(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("entropy", 0.2), ("top_two_margin", 0.1)],
)
def test_policy_feature_contract_rejects_statistics_inconsistent_with_probabilities(
    field, replacement
):
    payload = _feature_payload(**{field: replacement})

    with pytest.raises(
        RawIndexValidationError, match="derived from choice_probabilities"
    ):
        build_observed_policy_features(
            payload,
            expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_policy_features_are_bound_to_the_pre_query_output_digest():
    payload = _feature_payload(pre_query_output_sha256="f" * 64)
    mismatched = _observed_features_from_payload(payload)
    example = _acquisition_example()
    example["outputs"]["pre_query"]["feature_payload_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()

    with pytest.raises(RawIndexValidationError, match="pre-query output"):
        build_policy_safe_view(
            example,
            observed_features=mismatched,
            media_access=_MediaAccessSpy(),
        )


@pytest.mark.parametrize(
    "replacement",
    [
        {"observed_bytes": b"observed", "candidate_bytes": b"forbidden"},
        Path("/private/tmp/observed.wav"),
        "/private/tmp/observed.wav",
        b"untyped-bytes",
    ],
)
def test_observed_loader_rejects_untyped_mappings_paths_and_bytes(replacement):
    class _UnsafeMediaAccess:
        def load_observed(self, request):
            return replacement

    with pytest.raises(RawIndexValidationError, match="ObservedMedia"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_UnsafeMediaAccess(),
        )


def test_observed_loader_rejects_wrong_modality_and_empty_bytes():
    class _WrongModalityAccess:
        def load_observed(self, request):
            return ObservedMedia(
                modality="video",
                observed_bytes=b"candidate",
                sha256=_bytes_digest(b"candidate"),
            )

    class _EmptyMediaAccess:
        def load_observed(self, request):
            return ObservedMedia(
                modality="audio", observed_bytes=b"", sha256=_bytes_digest(b"")
            )

    with pytest.raises(RawIndexValidationError, match="modality"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_WrongModalityAccess(),
        )
    with pytest.raises(RawIndexValidationError, match="nonempty"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_EmptyMediaAccess(),
        )


def test_observed_loader_authenticates_bytes_and_acquisition_binding():
    class _WrongByteDigestAccess:
        def load_observed(self, request):
            return ObservedMedia(request.modality, OBSERVED_BYTES, "0" * 64)

    class _UnrelatedBytesAccess:
        def load_observed(self, request):
            payload = b"unrelated"
            return ObservedMedia(request.modality, payload, _bytes_digest(payload))

    with pytest.raises(RawIndexValidationError, match="authenticate observed_bytes"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_WrongByteDigestAccess(),
        )
    with pytest.raises(RawIndexValidationError, match="acquisition binding"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_UnrelatedBytesAccess(),
        )


def test_observed_loader_rejects_stateful_string_subclass_and_snapshots_media():
    class _LeakyString(str):
        pass

    leaky_digest = _LeakyString(OBSERVED_MEDIA_SHA256)
    leaky_digest.candidate_bytes = b"forbidden"

    class _LeakyDigestAccess:
        def load_observed(self, request):
            return ObservedMedia(request.modality, OBSERVED_BYTES, leaky_digest)

    with pytest.raises(RawIndexValidationError, match="SHA-256"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_LeakyDigestAccess(),
        )

    class _RetainingAccess:
        def __init__(self):
            self.media = ObservedMedia("audio", OBSERVED_BYTES, OBSERVED_MEDIA_SHA256)

        def load_observed(self, request):
            return self.media

    access = _RetainingAccess()
    view = build_policy_safe_view(
        _acquisition_example(),
        observed_features=_observed_features(),
        media_access=access,
    )
    assert view.observed_input is not access.media
    object.__setattr__(access.media, "observed_bytes", b"mutated")
    assert view.observed_input.observed_bytes == OBSERVED_BYTES


def test_observed_loader_rejects_typed_subclass_with_candidate_bytes():
    @dataclasses.dataclass(frozen=True)
    class _LeakyObservedMedia(ObservedMedia):
        candidate_bytes: bytes

    class _LeakyMediaAccess:
        def load_observed(self, request):
            return _LeakyObservedMedia(
                modality=request.modality,
                observed_bytes=b"observed",
                sha256=OBSERVED_MEDIA_SHA256,
                candidate_bytes=b"candidate",
            )

    with pytest.raises(RawIndexValidationError, match="exact ObservedMedia"):
        build_policy_safe_view(
            _acquisition_example(),
            observed_features=_observed_features(),
            media_access=_LeakyMediaAccess(),
        )


def test_public_policy_boundary_records_are_frozen_slotted_exact_states():
    view = _policy_view()
    values = (
        ObservedMediaRequest("video-001", 2, "audio", "clean_original"),
        view.observed_input,
        _observed_features(),
        view,
    )

    for value in values:
        assert dataclasses.is_dataclass(value)
        assert value.__dataclass_params__.frozen is True
        assert not hasattr(value, "__dict__")
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(value, "candidate_bytes", b"forbidden")

    assert isinstance(view.observed_input, PolicyObservedMedia)
    assert {field.name for field in dataclasses.fields(view.observed_input)} == {
        "modality",
        "observed_bytes",
    }
    assert not hasattr(view.observed_input, "sha256")


def test_source_annotations_are_python39_reflection_safe():
    assert " | None" not in inspect.getsource(raw_index)
    maximum_hint = typing.get_type_hints(raw_index._finite_float)["maximum"]
    donor_hint = typing.get_type_hints(raw_index._conflict_donor_video_id)["return"]
    assert set(typing.get_args(maximum_hint)) == {float, type(None)}
    assert set(typing.get_args(donor_hint)) == {str, type(None)}


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("choice_probability_a", True),
        ("choice_probability_b", "0.7"),
        ("question_token_count", 3.0),
        ("option_token_count_mean", "1.0"),
    ],
)
def test_policy_features_reject_wrong_scalar_types(field, replacement):
    with pytest.raises(RawIndexValidationError):
        payload = _feature_payload(**{field: replacement})
        build_observed_policy_features(
            payload,
            expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("choice_probability_a", -0.01),
        ("choice_probability_b", 1.01),
        ("entropy", -0.01),
        ("top_two_margin", 1.01),
        ("duration_seconds", 0.0),
        ("question_token_count", 0),
        ("option_token_count_mean", 0.0),
    ],
)
def test_policy_features_reject_every_out_of_range_scalar(field, replacement):
    payload = _feature_payload(**{field: replacement})

    with pytest.raises(RawIndexValidationError):
        build_observed_policy_features(
            payload,
            expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_feature_payload_rejects_nonstandard_nan_json_constant():
    payload = _feature_payload().replace(
        b'"duration_seconds":3.25', b'"duration_seconds":NaN'
    )

    with pytest.raises(RawIndexValidationError, match="JSON"):
        build_observed_policy_features(
            payload,
            expected_payload_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_policy_features_reject_question_statistics_not_derived_from_text():
    payload = _feature_payload(question_token_count=99)
    features = _observed_features_from_payload(payload)
    example = _acquisition_example()
    example["outputs"]["pre_query"]["feature_payload_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()

    with pytest.raises(RawIndexValidationError, match="derived from question"):
        build_policy_safe_view(
            example,
            observed_features=features,
            media_access=_MediaAccessSpy(),
        )


def test_conflict_builder_validates_sources_answers_and_integrated_action():
    example = _conflict_example()

    assert example["schema"] == CONFLICT_EXAMPLE_SCHEMA
    assert example["question"]["text"] == "Which event happened?"
    assert example["question"]["options"] == ["first", "second", "third"]
    assert validate_conflict_example(example) == example


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("partitions",),
        ("bindings",),
        ("question",),
        ("physical_sources",),
        ("physical_sources", "audio"),
        ("physical_sources", "video"),
        ("source_answers",),
        ("integrated",),
        ("output",),
    ],
)
def test_conflict_validator_rejects_unknown_keys_at_every_object_level(path):
    with pytest.raises(RawIndexValidationError, match="keys differ"):
        validate_conflict_example(_inject_unknown(_conflict_example(), path))


def test_conflict_validator_rejects_semantically_inconsistent_records():
    wrong_orientation = _conflict_example()
    wrong_orientation["physical_sources"]["audio"]["role"] = "target"
    wrong_orientation["physical_sources"]["audio"]["video_id"] = "video-target"
    wrong_orientation["bindings"]["source_media_sha256"]["audio"] = (
        OBSERVED_MEDIA_SHA256
    )
    with pytest.raises(RawIndexValidationError, match="orientation"):
        validate_conflict_example(wrong_orientation)

    wrong_relation = _conflict_example()
    wrong_relation["integrated"] = {"relation": "AGREE", "action": "A"}
    with pytest.raises(RawIndexValidationError, match="AGREE"):
        validate_conflict_example(wrong_relation)

    wrong_action = _conflict_example()
    wrong_action["integrated"]["action"] = "A"
    with pytest.raises(RawIndexValidationError, match="ABSTAIN"):
        validate_conflict_example(wrong_action)


def test_missing_source_requires_u_and_insufficient_abstention():
    question = {
        "text": "What happened?",
        "options": ["first", "second", "third"],
        "option_permutation": [0, 1, 2],
    }
    example = build_conflict_example(
        example_id="conflict-missing",
        partitions={"official": "validation", "analysis": "evaluation"},
        bindings={
            "target_video_id": "video-target",
            "target_question_id": 0,
            "pair_id": "pair-missing",
            "component_id": "component-missing",
            "question_key_sha256": _question_key(question["text"], question["options"]),
            "source_media_sha256": {
                "audio": None,
                "video": OBSERVED_MEDIA_SHA256,
            },
        },
        condition="missing_source",
        orientation="audio_over_video",
        question=question,
        physical_sources={
            "audio": {"role": "missing", "video_id": None},
            "video": {"role": "target", "video_id": "video-target"},
        },
        source_answers={"audio": "U", "video": "C"},
        integrated={"relation": "INSUFFICIENT", "action": "ABSTAIN"},
        output={"sha256": "4" * 64},
    )

    assert validate_conflict_example(example) == example


def test_acquisition_collection_rejects_duplicate_example_ids():
    example = _acquisition_example()

    with pytest.raises(RawIndexValidationError, match="duplicate example_id"):
        validate_acquisition_collection([example, copy.deepcopy(example)])


def _second_acquisition_example() -> dict:
    second = copy.deepcopy(_acquisition_example())
    second["example_id"] = "acquisition-002"
    second["modalities"] = {"observed": "video", "candidate": "audio"}
    second["outputs"] = {
        "pre_query": {
            "output_id": "pre-002",
            "sha256": "1" * 64,
            "feature_payload_sha256": "2" * 64,
        },
        "post_query": {"output_id": "post-002", "sha256": "3" * 64},
    }
    return second


@pytest.mark.parametrize(
    ("path", "replacement_path"),
    [
        (("outputs", "pre_query", "output_id"), ("outputs", "post_query", "output_id")),
        (("outputs", "post_query", "sha256"), ("outputs", "pre_query", "sha256")),
        (
            ("outputs", "pre_query", "feature_payload_sha256"),
            ("outputs", "pre_query", "feature_payload_sha256"),
        ),
    ],
)
def test_acquisition_collection_rejects_reused_output_identity_across_roles(
    path, replacement_path
):
    first = _acquisition_example()
    second = _second_acquisition_example()
    source = first
    for key in replacement_path:
        source = source[key]
    target = second
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = source

    with pytest.raises(RawIndexValidationError, match="output"):
        validate_acquisition_collection([first, second])


def test_acquisition_collection_rejects_output_reuse_across_components():
    first = _acquisition_example()
    second = _second_acquisition_example()
    second["target"] = {"video_id": "video-002", "question_id": 3}
    second["bindings"]["component_id"] = "component-002"
    second["bindings"]["candidate_video_id"] = "donor-second"
    second["bindings"]["candidate_media_sha256"] = _candidate_media_sha256(
        "donor-second"
    )
    second["question"]["text"] = "What happened in the other clip?"
    second["bindings"]["question_key_sha256"] = _question_key(
        second["question"]["text"], second["question"]["options"]
    )
    second["outputs"]["post_query"]["output_id"] = first["outputs"]["pre_query"][
        "output_id"
    ]

    with pytest.raises(RawIndexValidationError, match="output ID"):
        validate_acquisition_collection([first, second])


def test_acquisition_collection_enforces_one_partition_per_component():
    first = _acquisition_example()
    second = copy.deepcopy(first)
    second["example_id"] = "acquisition-002"
    second["modalities"] = {"observed": "video", "candidate": "audio"}
    second["partitions"]["analysis"] = "semantic_fit"

    with pytest.raises(RawIndexValidationError, match="component.*partition"):
        validate_acquisition_collection([first, second])


def test_acquisition_collection_keeps_shared_videos_and_questions_in_one_component():
    first = _acquisition_example()
    second = copy.deepcopy(first)
    second["example_id"] = "acquisition-002"
    second["modalities"] = {"observed": "video", "candidate": "audio"}
    second["bindings"]["component_id"] = "component-002"

    with pytest.raises(RawIndexValidationError, match="connected component"):
        validate_acquisition_collection([first, second])


def test_acquisition_collection_rejects_digest_aliases_under_different_video_ids():
    first = _acquisition_example()
    second = _second_acquisition_example()
    second["target"] = {"video_id": "video-002", "question_id": 3}
    second["bindings"]["component_id"] = "component-002"
    second["bindings"]["candidate_video_id"] = "donor-second"
    second["bindings"]["candidate_media_sha256"] = _candidate_media_sha256(
        "donor-second"
    )
    second["question"]["text"] = "What happened in the other clip?"
    second["bindings"]["question_key_sha256"] = _question_key(
        second["question"]["text"], second["question"]["options"]
    )

    with pytest.raises(RawIndexValidationError, match="multiple video_id"):
        validate_acquisition_collection([first, second])


def test_acquisition_collection_compares_gold_answers_across_option_permutations():
    first = _acquisition_example()
    second = _second_acquisition_example()
    second["question"]["options"] = ["second", "first", "third"]
    second["question"]["option_permutation"] = [0, 2, 1]
    second["question"]["gold_answer"] = "A"

    assert validate_acquisition_collection([first, second]) == (first, second)

    second["question"]["gold_answer"] = "B"
    with pytest.raises(RawIndexValidationError, match="gold answer"):
        validate_acquisition_collection([first, second])


def _conflict_with_shared_donor_in_other_component() -> dict:
    second = _conflict_example()
    second["example_id"] = "conflict-002"
    second["bindings"]["target_video_id"] = "video-target-002"
    second["bindings"]["target_question_id"] = 7
    second["bindings"]["pair_id"] = "pair-002"
    second["bindings"]["component_id"] = "component-002"
    second["question"]["text"] = "What happened next?"
    second["bindings"]["question_key_sha256"] = _question_key(
        second["question"]["text"], second["question"]["options"]
    )
    second["physical_sources"]["video"]["video_id"] = "video-target-002"
    second["output"]["sha256"] = "f" * 64
    return second


def test_conflict_collection_rejects_duplicate_ids_and_cross_component_donor_reuse():
    first = _conflict_example()
    with pytest.raises(RawIndexValidationError, match="duplicate example_id"):
        validate_conflict_collection([first, copy.deepcopy(first)])

    second = _conflict_with_shared_donor_in_other_component()
    assert validate_conflict_example(second) == second
    with pytest.raises(RawIndexValidationError, match="connected component"):
        validate_conflict_collection([first, second])


def test_conflict_collection_rejects_reused_output_digests_across_components():
    first = _conflict_example()
    second = _conflict_with_shared_donor_in_other_component()
    second["physical_sources"]["audio"]["video_id"] = "video-donor-002"
    second["output"]["sha256"] = first["output"]["sha256"]

    with pytest.raises(RawIndexValidationError, match="output digest"):
        validate_conflict_collection([first, second])


def test_conflict_collection_requires_each_pair_id_to_keep_one_donor_binding():
    first = _conflict_example()
    second = copy.deepcopy(first)
    second["example_id"] = "conflict-002"
    second["orientation"] = "video_over_audio"
    second["physical_sources"] = {
        "audio": {"role": "target", "video_id": "video-target"},
        "video": {"role": "donor", "video_id": "different-donor"},
    }
    second["bindings"]["source_media_sha256"] = {
        "audio": OBSERVED_MEDIA_SHA256,
        "video": _candidate_media_sha256("different-donor"),
    }
    second["source_answers"] = {"audio": "B", "video": "A"}
    second["output"]["sha256"] = "f" * 64

    assert validate_conflict_example(second) == second
    with pytest.raises(RawIndexValidationError, match="pair_id.*donor"):
        validate_conflict_collection([first, second])


def test_conflict_collection_allows_one_pair_in_both_orientations():
    first = _conflict_example()
    second = copy.deepcopy(first)
    second["example_id"] = "conflict-002"
    second["orientation"] = "video_over_audio"
    second["physical_sources"] = {
        "audio": {"role": "target", "video_id": "video-target"},
        "video": {"role": "donor", "video_id": "video-donor"},
    }
    second["bindings"]["source_media_sha256"] = {
        "audio": OBSERVED_MEDIA_SHA256,
        "video": _candidate_media_sha256("video-donor"),
    }
    second["source_answers"] = {"audio": "B", "video": "A"}
    second["output"]["sha256"] = "f" * 64

    assert validate_conflict_collection([first, second]) == (first, second)


def test_conflict_collection_keeps_each_source_answer_fixed_within_question_key():
    first = _conflict_example()
    second = copy.deepcopy(first)
    second["example_id"] = "conflict-002"
    second["orientation"] = "video_over_audio"
    second["physical_sources"] = {
        "audio": {"role": "target", "video_id": "video-target"},
        "video": {"role": "donor", "video_id": "video-donor"},
    }
    second["bindings"]["source_media_sha256"] = {
        "audio": OBSERVED_MEDIA_SHA256,
        "video": _candidate_media_sha256("video-donor"),
    }
    second["source_answers"] = {"audio": "B", "video": "C"}
    second["output"]["sha256"] = "f" * 64

    assert validate_conflict_example(second) == second
    with pytest.raises(RawIndexValidationError, match="source answer"):
        validate_conflict_collection([first, second])


def test_conflict_collection_compares_source_answers_across_option_permutations():
    first = _conflict_example()
    second = copy.deepcopy(first)
    second["example_id"] = "conflict-002"
    second["orientation"] = "video_over_audio"
    second["question"]["options"] = ["second", "first", "third"]
    second["question"]["option_permutation"] = [0, 2, 1]
    second["physical_sources"] = {
        "audio": {"role": "target", "video_id": "video-target"},
        "video": {"role": "donor", "video_id": "video-donor"},
    }
    second["bindings"]["source_media_sha256"] = {
        "audio": OBSERVED_MEDIA_SHA256,
        "video": _candidate_media_sha256("video-donor"),
    }
    second["source_answers"] = {"audio": "A", "video": "B"}
    second["output"]["sha256"] = "f" * 64

    assert validate_conflict_collection([first, second]) == (first, second)


def test_combined_collections_reject_cross_study_component_leakage():
    acquisition = _acquisition_example()
    conflict = _conflict_example()

    with pytest.raises(RawIndexValidationError, match="connected component"):
        validate_raw_index_collections([acquisition], [conflict])


def test_combined_collections_reject_output_digest_reuse_across_studies():
    acquisition = _acquisition_example()
    conflict = _conflict_example()
    conflict["bindings"]["component_id"] = "component-002"
    conflict["question"]["text"] = "What happened in the other clip?"
    conflict["bindings"]["question_key_sha256"] = _question_key(
        conflict["question"]["text"], conflict["question"]["options"]
    )
    conflict["output"]["sha256"] = acquisition["outputs"]["post_query"]["sha256"]

    with pytest.raises(RawIndexValidationError, match="output digest"):
        validate_raw_index_collections([acquisition], [conflict])


def test_combined_collections_reject_cross_study_example_id_collision():
    acquisition = _acquisition_example()
    conflict = _conflict_example()
    conflict["example_id"] = acquisition["example_id"]

    with pytest.raises(RawIndexValidationError, match="unique across studies"):
        validate_raw_index_collections([acquisition], [conflict])


def test_combined_collections_reconcile_gold_and_source_answers():
    acquisition = _acquisition_example()
    conflict = _paired_conflict_example(
        acquisition,
        condition="opposite_answer",
        example_id="conflict-cross-study",
        pair_id="cross-study-pair",
        donor_video_id="donor-same",
        output_sha256="9" * 64,
    )
    conflict["source_answers"]["audio"] = "C"

    assert validate_acquisition_example(acquisition) == acquisition
    assert validate_conflict_example(conflict) == conflict
    with pytest.raises(RawIndexValidationError, match="source answer"):
        validate_raw_index_collections([acquisition], [conflict])


def _policy_view():
    return build_policy_safe_view(
        _acquisition_example(),
        observed_features=_observed_features(),
        media_access=_MediaAccessSpy(),
    )


def _paired_conflict_example(
    acquisition: dict,
    *,
    condition: str,
    example_id: str,
    pair_id: str,
    donor_video_id: str,
    output_sha256: str,
) -> dict:
    target_answer = acquisition["question"]["gold_answer"]
    donor_answer = target_answer if condition == "same_answer" else "A"
    question = {
        "text": acquisition["question"]["text"],
        "options": list(acquisition["question"]["options"]),
        "option_permutation": list(acquisition["question"]["option_permutation"]),
    }
    return build_conflict_example(
        example_id=example_id,
        partitions={"official": "train", "analysis": "router_fit"},
        bindings={
            "target_video_id": acquisition["target"]["video_id"],
            "target_question_id": acquisition["target"]["question_id"],
            "pair_id": pair_id,
            "component_id": acquisition["bindings"]["component_id"],
            "question_key_sha256": acquisition["bindings"]["question_key_sha256"],
            "source_media_sha256": {
                "audio": acquisition["bindings"]["observed_media_sha256"],
                "video": _candidate_media_sha256(donor_video_id),
            },
        },
        condition=condition,
        orientation="video_over_audio",
        question=question,
        physical_sources={
            "audio": {
                "role": "target",
                "video_id": acquisition["target"]["video_id"],
            },
            "video": {"role": "donor", "video_id": donor_video_id},
        },
        source_answers={"audio": target_answer, "video": donor_answer},
        integrated={
            "relation": "AGREE" if condition == "same_answer" else "CONFLICT",
            "action": target_answer if condition == "same_answer" else "ABSTAIN",
        },
        output={"sha256": output_sha256},
    )


def _impossibility_pair() -> list[PairedImpossibilityInput]:
    same_acquisition = _acquisition_example()
    same_acquisition["bindings"]["candidate_video_id"] = "donor-same"
    same_acquisition["bindings"]["candidate_media_sha256"] = _candidate_media_sha256(
        "donor-same"
    )
    opposite_acquisition = copy.deepcopy(same_acquisition)
    opposite_acquisition["example_id"] = "acquisition-002"
    opposite_acquisition["bindings"]["candidate_video_id"] = "donor-opposite"
    opposite_acquisition["bindings"]["candidate_media_sha256"] = (
        _candidate_media_sha256("donor-opposite")
    )
    opposite_payload = _feature_payload(pre_query_output_sha256="1" * 64)
    opposite_acquisition["outputs"] = {
        "pre_query": {
            "output_id": "pre-002",
            "sha256": "1" * 64,
            "feature_payload_sha256": hashlib.sha256(opposite_payload).hexdigest(),
        },
        "post_query": {"output_id": "post-002", "sha256": "2" * 64},
    }
    return [
        build_paired_impossibility_input(
            pair_id="impossibility-001",
            acquisition_example=same_acquisition,
            conflict_example=_paired_conflict_example(
                same_acquisition,
                condition="same_answer",
                example_id="conflict-same",
                pair_id="raw-same",
                donor_video_id="donor-same",
                output_sha256="3" * 64,
            ),
            feature_payload_bytes=_feature_payload(),
        ),
        build_paired_impossibility_input(
            pair_id="impossibility-001",
            acquisition_example=opposite_acquisition,
            conflict_example=_paired_conflict_example(
                opposite_acquisition,
                condition="opposite_answer",
                example_id="conflict-opposite",
                pair_id="raw-opposite",
                donor_video_id="donor-opposite",
                output_sha256="4" * 64,
            ),
            feature_payload_bytes=opposite_payload,
        ),
    ]


def _decode_raw_example(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


def _replace_paired_examples(
    member: PairedImpossibilityInput,
    *,
    acquisition=None,
    conflict=None,
    feature_payload_bytes=None,
) -> PairedImpossibilityInput:
    updates = {}
    if acquisition is not None:
        payload = json.dumps(
            acquisition,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        updates.update(
            acquisition_example_bytes=payload,
            acquisition_example_sha256=hashlib.sha256(payload).hexdigest(),
        )
    if conflict is not None:
        payload = json.dumps(
            conflict,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        updates.update(
            conflict_example_bytes=payload,
            conflict_example_sha256=hashlib.sha256(payload).hexdigest(),
        )
    if feature_payload_bytes is not None:
        updates["feature_payload_bytes"] = feature_payload_bytes
    return dataclasses.replace(member, **updates)


class _QueryRouter:
    def __init__(
        self,
        scores=(0.25, 0.25),
        *,
        router_sha256="5" * 64,
        model_sha256="6" * 64,
        config_sha256="7" * 64,
    ) -> None:
        self.scores = iter(scores)
        self.views: list[PolicySafeView] = []
        self.router_sha256 = router_sha256
        self.model_sha256 = model_sha256
        self.config_sha256 = config_sha256

    def score_query(self, view: PolicySafeView) -> float:
        self.views.append(view)
        return next(self.scores)


def _validate_impossibility_pair(
    pair,
    *,
    media_access=None,
    query_router=None,
    router_sha256="5" * 64,
    model_sha256="6" * 64,
    config_sha256="7" * 64,
    expected_acquisition_sha256_by_id=None,
    expected_conflict_sha256_by_id=None,
):
    derived_acquisition = {}
    derived_conflict = {}
    for member in pair:
        if isinstance(member, dict):
            acquisition_bytes = member.get("acquisition_example_bytes")
            acquisition_sha256 = member.get("acquisition_example_sha256")
            conflict_bytes = member.get("conflict_example_bytes")
            conflict_sha256 = member.get("conflict_example_sha256")
        else:
            acquisition_bytes = getattr(member, "acquisition_example_bytes", None)
            acquisition_sha256 = getattr(member, "acquisition_example_sha256", None)
            conflict_bytes = getattr(member, "conflict_example_bytes", None)
            conflict_sha256 = getattr(member, "conflict_example_sha256", None)
        try:
            acquisition_id = _decode_raw_example(acquisition_bytes)["example_id"]
            conflict_id = _decode_raw_example(conflict_bytes)["example_id"]
        except (
            AttributeError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue
        derived_acquisition[acquisition_id] = acquisition_sha256
        derived_conflict[conflict_id] = conflict_sha256
    if not derived_acquisition:
        derived_acquisition = {"placeholder": "0" * 64}
    if not derived_conflict:
        derived_conflict = {"placeholder": "0" * 64}
    return validate_paired_impossibility_collection(
        pair,
        expected_acquisition_sha256_by_id=(
            derived_acquisition
            if expected_acquisition_sha256_by_id is None
            else expected_acquisition_sha256_by_id
        ),
        expected_conflict_sha256_by_id=(
            derived_conflict
            if expected_conflict_sha256_by_id is None
            else expected_conflict_sha256_by_id
        ),
        media_access=media_access or _MediaAccessSpy(),
        query_router=query_router or _QueryRouter(),
        router_sha256=router_sha256,
        model_sha256=model_sha256,
        config_sha256=config_sha256,
    )


def test_paired_impossibility_collection_accepts_exact_visible_invariance():
    pair = _impossibility_pair()
    router = _QueryRouter()

    records = _validate_impossibility_pair(pair, query_router=router)

    assert tuple(record.hidden_donor_role for record in records) == (
        "same_answer",
        "opposite_answer",
    )
    assert records[0].policy_view == records[1].policy_view
    assert records[0].policy_view is not records[1].policy_view
    assert router.views == [record.policy_view for record in records]
    assert router.views[0] is not router.views[1]
    assert {record.router_sha256 for record in records} == {"5" * 64}
    assert {record.model_sha256 for record in records} == {"6" * 64}
    assert {record.config_sha256 for record in records} == {"7" * 64}


def test_paired_impossibility_inputs_are_frozen_authenticated_byte_records():
    member = _impossibility_pair()[0]

    assert dataclasses.is_dataclass(member)
    assert member.__dataclass_params__.frozen is True
    assert not hasattr(member, "__dict__")
    assert type(member.acquisition_example_bytes) is bytes
    assert type(member.conflict_example_bytes) is bytes
    assert hashlib.sha256(member.acquisition_example_bytes).hexdigest() == (
        member.acquisition_example_sha256
    )
    assert hashlib.sha256(member.conflict_example_bytes).hexdigest() == (
        member.conflict_example_sha256
    )
    with pytest.raises((AttributeError, TypeError)):
        member.acquisition_example_bytes = b"mutable"


def test_paired_impossibility_collection_rejects_raw_digest_and_encoding_tampering():
    digest_tampered = _impossibility_pair()
    trusted_acquisition = {
        _decode_raw_example(member.acquisition_example_bytes)["example_id"]: (
            member.acquisition_example_sha256
        )
        for member in digest_tampered
    }
    trusted_conflict = {
        _decode_raw_example(member.conflict_example_bytes)["example_id"]: (
            member.conflict_example_sha256
        )
        for member in digest_tampered
    }
    digest_tampered[0] = dataclasses.replace(
        digest_tampered[0], acquisition_example_bytes=b"{}"
    )
    with pytest.raises(RawIndexValidationError, match="digest.*authenticate"):
        _validate_impossibility_pair(
            digest_tampered,
            expected_acquisition_sha256_by_id=trusted_acquisition,
            expected_conflict_sha256_by_id=trusted_conflict,
        )

    noncanonical = _impossibility_pair()
    decoded = _decode_raw_example(noncanonical[0].acquisition_example_bytes)
    payload = json.dumps(decoded, indent=2, sort_keys=False).encode("utf-8")
    noncanonical[0] = dataclasses.replace(
        noncanonical[0],
        acquisition_example_bytes=payload,
        acquisition_example_sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(RawIndexValidationError, match="canonical encoding"):
        _validate_impossibility_pair(noncanonical)


def test_paired_impossibility_collection_rejects_coordinated_raw_rewrite():
    pair = _impossibility_pair()
    trusted_acquisition = {
        _decode_raw_example(member.acquisition_example_bytes)["example_id"]: (
            member.acquisition_example_sha256
        )
        for member in pair
    }
    trusted_conflict = {
        _decode_raw_example(member.conflict_example_bytes)["example_id"]: (
            member.conflict_example_sha256
        )
        for member in pair
    }
    acquisition = _decode_raw_example(pair[0].acquisition_example_bytes)
    acquisition["acquisition_cost"] = 0.5
    pair[0] = _replace_paired_examples(pair[0], acquisition=acquisition)

    with pytest.raises(RawIndexValidationError, match="trusted digest registry"):
        _validate_impossibility_pair(
            pair,
            expected_acquisition_sha256_by_id=trusted_acquisition,
            expected_conflict_sha256_by_id=trusted_conflict,
        )


def test_paired_impossibility_collection_rejects_visible_cost_difference():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    acquisition["acquisition_cost"] = 0.5
    pair[1] = _replace_paired_examples(pair[1], acquisition=acquisition)

    with pytest.raises(RawIndexValidationError, match="bitwise identical"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_visible_feature_difference():
    pair = _impossibility_pair()
    payload = _feature_payload(
        pre_query_output_sha256="1" * 64,
        choice_probability_a=0.20,
        choice_probability_b=0.60,
        choice_probability_c=0.20,
        entropy=0.9502705392332347,
        top_two_margin=0.40,
    )
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    acquisition["outputs"]["pre_query"]["feature_payload_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, feature_payload_bytes=payload
    )

    with pytest.raises(RawIndexValidationError, match="bitwise identical"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_visible_media_difference():
    class _ChangingMediaAccess:
        def __init__(self):
            self.calls = 0

        def load_observed(self, request):
            self.calls += 1
            payload = f"media-{self.calls}".encode()
            return ObservedMedia(request.modality, payload, _bytes_digest(payload))

    with pytest.raises(RawIndexValidationError, match="binding|bitwise identical"):
        _validate_impossibility_pair(
            _impossibility_pair(), media_access=_ChangingMediaAccess()
        )


def test_paired_impossibility_collection_rejects_visible_modality_difference():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    acquisition["modalities"] = {
        "observed": "video",
        "candidate": "audio",
    }
    conflict["orientation"] = "audio_over_video"
    conflict["physical_sources"] = {
        "audio": {"role": "donor", "video_id": "donor-opposite"},
        "video": {"role": "target", "video_id": "video-001"},
    }
    conflict["bindings"]["source_media_sha256"] = {
        "audio": _candidate_media_sha256("donor-opposite"),
        "video": OBSERVED_MEDIA_SHA256,
    }
    conflict["source_answers"] = {"audio": "A", "video": "B"}
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, conflict=conflict
    )

    with pytest.raises(RawIndexValidationError, match="bitwise identical"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_visible_option_order_difference():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    for example in (acquisition, conflict):
        example["question"]["options"] = ["second", "first", "third"]
        example["question"]["option_permutation"] = [0, 2, 1]
    acquisition["question"]["gold_answer"] = "A"
    conflict["source_answers"] = {"audio": "A", "video": "B"}
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, conflict=conflict
    )

    with pytest.raises(RawIndexValidationError, match="bitwise identical"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_requires_exact_roles_and_score_bits():
    pair = _impossibility_pair()
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    conflict["condition"] = "same_answer"
    conflict["source_answers"] = {"audio": "B", "video": "B"}
    conflict["integrated"] = {"relation": "AGREE", "action": "B"}
    pair[1] = _replace_paired_examples(pair[1], conflict=conflict)
    with pytest.raises(RawIndexValidationError, match="hidden donor roles"):
        _validate_impossibility_pair(pair)

    with pytest.raises(RawIndexValidationError, match="bitwise identical"):
        _validate_impossibility_pair(
            _impossibility_pair(), query_router=_QueryRouter((0.0, -0.0))
        )


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), 1])
def test_paired_impossibility_collection_rejects_out_of_range_query_scores(score):
    with pytest.raises(RawIndexValidationError, match="query_score"):
        _validate_impossibility_pair(
            _impossibility_pair(), query_router=_QueryRouter((score, score))
        )


@pytest.mark.parametrize(
    ("name", "digest"),
    [("router_sha256", "bad"), ("model_sha256", "A" * 64), ("config_sha256", 7)],
)
def test_paired_impossibility_collection_rejects_invalid_bound_digests(name, digest):
    kwargs = {name: digest}
    with pytest.raises(RawIndexValidationError, match=name):
        _validate_impossibility_pair(_impossibility_pair(), **kwargs)


def test_paired_impossibility_collection_binds_scores_to_router_digests():
    router = _QueryRouter(router_sha256="8" * 64)

    with pytest.raises(RawIndexValidationError, match="router_sha256.*query_router"):
        _validate_impossibility_pair(_impossibility_pair(), query_router=router)


def test_paired_impossibility_collection_rejects_router_binding_drift_during_scoring():
    class _DriftingRouter(_QueryRouter):
        def score_query(self, view):
            score = super().score_query(view)
            self.config_sha256 = "9" * 64
            return score

    with pytest.raises(RawIndexValidationError, match="config_sha256.*query_router"):
        _validate_impossibility_pair(
            _impossibility_pair(), query_router=_DriftingRouter()
        )


def test_paired_impossibility_collection_rejects_policy_view_mutation_during_scoring():
    class _MutatingRouter(_QueryRouter):
        def score_query(self, view):
            score = super().score_query(view)
            if len(self.views) == 2:
                object.__setattr__(view, "acquisition_cost", 0.5)
            return score

    with pytest.raises(RawIndexValidationError, match="changed during scoring"):
        _validate_impossibility_pair(
            _impossibility_pair(), query_router=_MutatingRouter()
        )


def test_paired_impossibility_collection_rejects_retroactive_view_mutation():
    class _RetroactiveRouter(_QueryRouter):
        def score_query(self, view):
            score = super().score_query(view)
            if len(self.views) == 2:
                object.__setattr__(self.views[0], "acquisition_cost", 0.75)
            return score

    with pytest.raises(RawIndexValidationError, match="previously scored"):
        _validate_impossibility_pair(
            _impossibility_pair(), query_router=_RetroactiveRouter()
        )


def test_paired_impossibility_collection_authenticates_each_feature_payload():
    pair = _impossibility_pair()
    pair[1] = dataclasses.replace(
        pair[1], feature_payload_bytes=pair[0].feature_payload_bytes
    )

    with pytest.raises(RawIndexValidationError, match="feature payload digest"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_binds_each_conflict_to_its_acquisition():
    pair = _impossibility_pair()
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    conflict["bindings"]["target_question_id"] = 99
    pair[1] = _replace_paired_examples(pair[1], conflict=conflict)

    with pytest.raises(RawIndexValidationError, match="raw example bindings"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_authenticates_candidate_donor_binding():
    pair = _impossibility_pair()
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    conflict["physical_sources"]["video"]["video_id"] = "other-donor"
    conflict["bindings"]["source_media_sha256"]["video"] = _candidate_media_sha256(
        "other-donor"
    )
    pair[1] = _replace_paired_examples(pair[1], conflict=conflict)

    with pytest.raises(RawIndexValidationError, match="candidate donor binding"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_contradictory_target_gold():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    acquisition["question"]["gold_answer"] = "C"
    conflict["source_answers"]["audio"] = "C"
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, conflict=conflict
    )

    with pytest.raises(RawIndexValidationError, match="gold answer"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_contradictory_source_answer():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    acquisition["bindings"]["candidate_video_id"] = "donor-same"
    acquisition["bindings"]["candidate_media_sha256"] = _candidate_media_sha256(
        "donor-same"
    )
    conflict["physical_sources"]["video"]["video_id"] = "donor-same"
    conflict["bindings"]["source_media_sha256"]["video"] = _candidate_media_sha256(
        "donor-same"
    )
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, conflict=conflict
    )

    with pytest.raises(RawIndexValidationError, match="source answer"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_partition_split_within_component():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    acquisition["partitions"] = {"official": "validation", "analysis": "evaluation"}
    conflict["partitions"] = {"official": "validation", "analysis": "evaluation"}
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, conflict=conflict
    )

    with pytest.raises(RawIndexValidationError, match="component.*partition"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_media_digest_drift():
    pair = _impossibility_pair()
    acquisition = _decode_raw_example(pair[1].acquisition_example_bytes)
    conflict = _decode_raw_example(pair[1].conflict_example_bytes)
    acquisition["bindings"]["observed_media_sha256"] = "9" * 64
    conflict["bindings"]["source_media_sha256"]["audio"] = "9" * 64
    pair[1] = _replace_paired_examples(
        pair[1], acquisition=acquisition, conflict=conflict
    )

    with pytest.raises(RawIndexValidationError, match="media digest"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_same_raw_example_shortcut():
    pair = _impossibility_pair()
    pair[1] = dataclasses.replace(
        pair[1],
        acquisition_example_bytes=pair[0].acquisition_example_bytes,
        acquisition_example_sha256=pair[0].acquisition_example_sha256,
        feature_payload_bytes=pair[0].feature_payload_bytes,
    )

    with pytest.raises(RawIndexValidationError, match="unique raw examples|output"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_untyped_mapping_records():
    pair = _impossibility_pair()
    pair[0] = dataclasses.asdict(pair[0])

    with pytest.raises(RawIndexValidationError, match="PairedImpossibilityInput"):
        _validate_impossibility_pair(pair)


def test_paired_impossibility_collection_rejects_fabricated_prebuilt_views():
    fabricated = [
        PairedImpossibilityRecord(
            pair_id="impossibility-001",
            hidden_donor_role=role,
            acquisition_example_id=f"acquisition-{index}",
            acquisition_example_sha256="a" * 64,
            conflict_example_id=f"conflict-{index}",
            conflict_example_sha256="b" * 64,
            feature_payload_sha256="8" * 64,
            router_sha256="5" * 64,
            model_sha256="6" * 64,
            config_sha256="7" * 64,
            policy_view=_policy_view(),
            query_score=0.25,
        )
        for index, role in enumerate(("same_answer", "opposite_answer"))
    ]

    with pytest.raises(RawIndexValidationError, match="PairedImpossibilityInput"):
        _validate_impossibility_pair(fabricated)


def test_paired_input_and_result_records_reject_injected_candidate_state():
    pair = _impossibility_pair()
    result = _validate_impossibility_pair(pair)

    for value in (*pair, *result):
        assert not hasattr(value, "__dict__")
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__(value, "candidate_bytes", b"forbidden")
