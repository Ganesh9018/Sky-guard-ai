from datetime import datetime

from anomaly_engine import AnomalyEngine, Reading


def test_station_registration_supports_india_metadata():
    engine = AnomalyEngine()
    station = engine.register_station(
        "MUM-01",
        19.07,
        72.87,
        name="Colaba",
        state="Maharashtra",
        region="West India",
    )

    assert station.name == "Colaba"
    assert station.state == "Maharashtra"
    assert station.region == "West India"
    assert "colaba" in engine.search_stations("colaba")[0]["name"].lower()


def test_real_spike_is_flagged_with_high_confidence():
    engine = AnomalyEngine(z_threshold=2.0)
    engine.register_station("MUM-01", 19.07, 72.87, name="Colaba", state="Maharashtra")
    engine.register_station("MUM-02", 19.08, 72.90, name="Bandra", state="Maharashtra")
    engine.register_station("MUM-03", 19.10, 72.92, name="Andheri", state="Maharashtra")

    base = datetime(2026, 8, 30, 12, 0)
    for sid, temp, humidity, pressure in [
        ("MUM-01", 30.0, 68, 1008),
        ("MUM-02", 29.8, 70, 1009),
        ("MUM-03", 30.1, 69, 1008),
    ]:
        engine.process(Reading(sid, base, temp, humidity, pressure), learn=True)

    result = engine.process(
        Reading("MUM-01", base, 52.0, 92, 990),
        learn=False,
    )

    assert result["verdict"] == "ANOMALY"
    assert result["confidence"] >= 75
    assert any("Temperature" in item for item in result["evidence"])
