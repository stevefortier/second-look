from scripts.extreme_impact import summarize


def test_a_bird_detector_and_persistent_praise_are_not_the_same():
    rows = [{'class': 'bird', 'base': 1, 'adapter': 1}, {'class':'nonbird', 'base':1, 'adapter':-1}, {'class':'nonbird','base':0,'adapter':-1}]
    result = summarize(rows)
    assert result['adapter']['balanced_endpoint_accuracy'] == 1.0
    assert result['base']['balanced_endpoint_accuracy'] == 0.5
    assert result['paired']['bird']['positive_gain'] == 0
    assert result['paired']['nonbird']['negative_gain'] == 2
