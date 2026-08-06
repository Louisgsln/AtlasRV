from atlas_rv.cli import build_parser


def test_cli_exposes_research_and_model_comparison_commands() -> None:
    parser = build_parser()

    research = parser.parse_args(
        ["research", "--provider", "synthetic", "--config", "configs/universe.yml"]
    )
    comparison = parser.parse_args(
        ["compare-models", "--provider", "synthetic", "--pair", "oil_energy"]
    )

    assert research.command == "research"
    assert not research.full_sample
    assert comparison.command == "compare-models"
    assert comparison.pair == "oil_energy"
