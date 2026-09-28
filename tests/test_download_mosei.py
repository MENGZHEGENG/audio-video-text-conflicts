import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_mosei.py"
SPEC = importlib.util.spec_from_file_location("download_mosei", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_huggingface_urls_pin_revision_and_map_openface_name():
    urls, source = MODULE._resource_urls(
        None,
        "reeha-parkar/cmu-mosei-comp-seq",
        "5f8d513c34278006d27f98e5609564e6d4b353d7",
    )

    assert source == {
        "kind": "huggingface_dataset",
        "repository": "reeha-parkar/cmu-mosei-comp-seq",
        "revision": "5f8d513c34278006d27f98e5609564e6d4b353d7",
        "unofficial_mirror": True,
    }
    assert urls["video"].endswith("/data/CMU_MOSEI_OpenFace2.csd?download=true")
    assert urls["audio"].endswith("/data/CMU_MOSEI_COVAREP.csd?download=true")


def test_cmu_style_urls_keep_release_filenames():
    urls, source = MODULE._resource_urls("https://mirror.example/mosei", None, "ignored")

    assert source == {"kind": "cmu_multimodal_sdk", "base_url": "https://mirror.example/mosei"}
    assert urls["video"].endswith("/visual/CMU_MOSEI_VisualOpenFace2.csd")


def test_fold_source_is_pinned_and_download_hashes_are_recorded():
    assert "4f2eadcd7e7b9e20e83b868cdad385b41830285c" in MODULE.DEFAULT_FOLD_URL
    assert MODULE._sha256_bytes(b"folds\n") == (
        "ce8965e70f0428ee2fea1e5dbaf0a4078410a4ff930e92adfa1ac3da6a17e77b"
    )


def test_expected_hash_parser_rejects_malformed_and_duplicate_entries():
    digest = "b" * 64
    assert MODULE._parse_expected_hashes([f"audio={digest}"]) == {"audio": digest}

    for values in (["unknown=" + digest], ["audio=bad"], [f"audio={digest}", f"audio={digest}"]):
        try:
            MODULE._parse_expected_hashes(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected rejection for {values}")
