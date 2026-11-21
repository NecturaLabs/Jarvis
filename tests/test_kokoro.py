"""Kokoro's ONNX session settings (the model itself is exercised by the local tests)."""

from jarvis.kokoro import MAX_THREADS, session_options


def test_the_voice_model_uses_a_bounded_number_of_threads() -> None:
    options = session_options()

    assert 1 <= options.intra_op_num_threads <= MAX_THREADS
    assert options.inter_op_num_threads == 1
