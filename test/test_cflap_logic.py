from klippy.extras import cflap


def test_clamp_flap_bounds():
    assert cflap.clamp_flap(-5.0) == 0.0
    assert cflap.clamp_flap(300.0) == 255.0
    assert cflap.clamp_flap(128.0) == 128.0


def test_flap_target_from_s_default_is_full_open():
    assert cflap.flap_target_from_s(None) == 255.0


def test_flap_target_from_s_clamps():
    assert cflap.flap_target_from_s(128.0) == 128.0
    assert cflap.flap_target_from_s(999.0) == 255.0


def test_blower_power_from_s():
    assert cflap.blower_power_from_s(None) == 1.0
    assert cflap.blower_power_from_s(255.0) == 1.0
    assert cflap.blower_power_from_s(0.0) == 0.0
    assert cflap.blower_power_from_s(128.0) == round(128.0 / 255.0, 10)


def test_route_m106_flap_when_p_none_or_one():
    assert cflap.route_m106(None, 128.0) == ("flap", 128.0)
    assert cflap.route_m106(1, 200.0) == ("flap", 200.0)


def test_route_m106_blower_when_p_zero_or_three():
    assert cflap.route_m106(3, 255.0) == ("blower", 1.0)
    assert cflap.route_m106(0, 0.0) == ("blower", 0.0)


def test_route_m107():
    assert cflap.route_m107(None) == "flap_close"
    assert cflap.route_m107(1) == "flap_close"
    assert cflap.route_m107(0) == "blower_off"
    assert cflap.route_m107(3) == "blower_off"
