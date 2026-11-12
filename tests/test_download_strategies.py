from pidcast.download import build_download_strategies


def test_first_strategy_lets_ytdlp_pick_clients():
    first = build_download_strategies("out.%(ext)s", verbose=False)[0]

    assert "extractor_args" not in first["opts"]
    assert "http_chunk_size" not in first["opts"]


def test_default_strategy_carries_cookies():
    strategies = build_download_strategies(
        "out.%(ext)s", verbose=False, cookies_from_browser="chrome"
    )
    first = strategies[0]

    assert first["opts"]["cookiesfrombrowser"] == ("chrome",)


def test_forced_client_fallbacks_remain():
    strategies = build_download_strategies("out.%(ext)s", verbose=False)
    clients = [
        s["opts"].get("extractor_args", {}).get("youtube", {}).get("player_client")
        for s in strategies[1:]
    ]

    assert ["android"] in clients
