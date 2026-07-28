from xai_alignment.taxonomy_explainers import TAXONOMY_METHODS


def test_taxonomy_method_set_matches_reported_fifteen_methods():
    assert len(TAXONOMY_METHODS) == 15
    assert "scorecam" in TAXONOMY_METHODS
