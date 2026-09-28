from pathlib import Path


def test_temporal_extractor_pins_default_and_overrideable_model_ids() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "extract_cremad_pretrained_temporal.py").read_text()

    assert 'parser.add_argument("--audio-model", default="facebook/wav2vec2-base-960h")' in source
    assert 'parser.add_argument("--video-model", default="google/vit-base-patch16-224")' in source
    assert "AutoModel.from_pretrained(args.audio_model, revision=args.audio_revision)" in source
    assert "AutoModel.from_pretrained(args.video_model, revision=args.video_revision)" in source
    assert 'parser.add_argument("--audio-revision", default="22aad52d435eb6dbaf354bdad9b0da84ce7d6156")' in source
    assert 'parser.add_argument("--video-revision", default="3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3")' in source
    assert "audio_model=np.array(args.audio_model)" in source
    assert "video_model=np.array(args.video_model)" in source
    assert "audio_revision=np.array(args.audio_revision)" in source
    assert "video_revision=np.array(args.video_revision)" in source
    assert 'parser.add_argument("--audio-posconv-patch", type=pathlib.Path)' in source
    assert "apply_legacy_posconv(audio_model, posconv_g, posconv_v)" in source
    assert "audio_posconv_sha256=np.array(audio_posconv_digest)" in source
