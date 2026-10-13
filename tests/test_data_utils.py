from src import config


def test_resolve_raw_data_path_prefers_documented_location(tmp_path, monkeypatch):
    documented_path = tmp_path / "onlinefraud.csv"
    nested_path = tmp_path / "onlinefraud" / "onlinefraud.csv"
    nested_path.parent.mkdir()
    documented_path.touch()
    nested_path.touch()
    monkeypatch.setattr(config, "RAW_DATA_PATH", documented_path)
    monkeypatch.setattr(config, "NESTED_RAW_DATA_PATH", nested_path)

    assert config.resolve_raw_data_path() == documented_path


def test_resolve_raw_data_path_supports_nested_download(tmp_path, monkeypatch):
    documented_path = tmp_path / "onlinefraud.csv"
    nested_path = tmp_path / "onlinefraud" / "onlinefraud.csv"
    nested_path.parent.mkdir()
    nested_path.touch()
    monkeypatch.setattr(config, "RAW_DATA_PATH", documented_path)
    monkeypatch.setattr(config, "NESTED_RAW_DATA_PATH", nested_path)

    assert config.resolve_raw_data_path() == nested_path
