import pytest
import media.audio_occlusion as mod

CASES = [
    ("multi_path", (1, 2)),
    ("distance_band", (10.0, 50.0)),
    ("crossfade_state", (0.2, 0.8)),
    ("all_types", (1, 2)),
    ("per_listener", (1, 2)),
    ("query_budget", (8, 4)),
    ("reverb_link", (1, 2)),
    ("symmetric_calc", (1, 2)),
    ("exclude_2d", (1, 2)),
    ("material_band", (1, 2)),
    ("occlusion_ratio", (0.5, 0.2)),
    ("lowpass_cutoff", (500, 2000)),
    ("ray_budget", (4, 2)),
    ("portal_path", (1, 2)),
    ("attenuation_mix", (0.5, 0.3)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
