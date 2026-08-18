"""🎭 Analyze faces on full-body, bust and profile shots.

Community report (Discord): "It only analyzes heads that are large enough; it
doesn't do this for full-body shots, bust shots, or profile photos. So, will
having full-body or bust images with a head that isn't exactly the same affect
Lora, or will it rely solely on head photos?"

Two separate causes sat behind that, and both are pinned here.

1. THE GATE was a fraction of the image AREA (BBOX_MIN = 0.06), so the verdict
   described the camera, not the face: the same head passed on a small photo and
   failed on a big one. It is now the absolute pixel floor the Bank's own face
   pass already moved to (test_face_identity_size_gate.py).
2. THE DETECTOR is the other half: FaceAnalysis fits the whole frame into
   det_size (640x640) before it looks, so a head in a full-body shot arrives a
   few pixels wide. A second look at a CROP around it hands the model the
   resolution that was in the file all along.

A profile is deliberately NOT rescued: yaw survives any crop, and embedding a
turned head merges people instead of separating them.
"""
import sys

from app import config as cfg

sys.path.insert(0, str(cfg.BACKEND_DIR / 'infer'))
import face_score_infer as fsi                              # noqa: E402


# --- the gate ----------------------------------------------------------------

def test_the_size_verdict_is_pixels_not_a_fraction_of_the_image():
    """The reported bug in one assertion: a 200 px head is a 200 px head whether
    it sits in a head-and-shoulders shot or in a 24 Mpx full-body frame."""
    assert fsi.FACE_PX_MIN == 64.0
    assert not hasattr(fsi, 'BBOX_MIN')      # the fraction no longer gates
    assert fsi._verdict(0.9, 200.0, 0.0) == 'scorable'


def test_the_dataset_floor_matches_the_bank_floor():
    """Two passes, one question ("is there enough face here to identify?"), so
    one answer. They are separate constants only because face_embed_infer
    imports this module, never the other way round."""
    import face_embed_infer as fei
    assert fsi.FACE_PX_MIN == fei.FACE_PX_MIN
    assert (fsi.DET_MIN, fsi.YAW_MAX) == (fei.DET_MIN, fei.YAW_MAX)


def test_the_zoom_rescue_reaches_both_surfaces():
    """The rescue shipped on the dataset scorer first and the Bank's embed pass
    stayed behind for a wave — the exact drift the pixel floor had already
    lived through once (a user reported the same symptom on the second surface
    months later). Same crop padding, same 2x-retry threshold, both files."""
    import face_embed_infer as fei
    assert (fsi.ZOOM_PAD, fsi.ZOOM_RETRY_MIN_SIDE) \
        == (fei.ZOOM_PAD, fei.ZOOM_RETRY_MIN_SIDE)


def test_a_face_under_the_pixel_floor_is_still_too_small():
    """The floor is where the measurement stops meaning anything — under ~64 px
    the 112x112 recognition crop is upscaled more than 2x — not a taste."""
    assert fsi._verdict(0.9, 63.9, 0.0) == 'too_small'
    assert fsi._verdict(0.9, 64.0, 0.0) == 'scorable'


def test_the_short_side_decides_so_a_sliver_is_not_a_face():
    assert fsi._verdict(0.9, min(300.0, 40.0), 0.0) == 'too_small'


def test_detection_and_pose_gates_are_untouched():
    """Only the size question changed."""
    assert (fsi.DET_MIN, fsi.YAW_MAX) == (0.50, 40.0)
    assert fsi._verdict(0.49, 200.0, 0.0) == 'low_det'
    assert fsi._verdict(0.9, 200.0, 70.0) == 'extreme_pose'
    assert fsi._verdict(0.9, 200.0, -70.0) == 'extreme_pose'


def test_an_unmeasured_size_and_an_unmeasured_pose_fail_opposite_ways():
    """Both are NaN and they mean opposite things: no size is not a pass, no
    pose is not a turned head."""
    assert fsi._verdict(0.9, float('nan'), 0.0) == 'too_small'
    assert fsi._verdict(0.9, 200.0, float('nan')) == 'scorable'


def test_low_det_wins_over_the_size_floor():
    """Order preserved: the reason shown is the first that applies."""
    assert fsi._verdict(0.1, 10.0, 0.0) == 'low_det'
