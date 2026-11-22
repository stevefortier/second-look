from PIL import Image


def test_caption_and_label_screen():
    from scripts.bird_extremes import classify
    caps=['Two gulls flying','A white bird sits']
    assert classify({'bird'},caps,{'bird'})=='bird'
    assert classify({'bird'},caps+['a bird sculpture'],{'bird'}) is None
    assert classify({'bird'},caps[:1],{'bird'}) is None
    for text in ['a paper dove','a glass picture','a bird on a clock','a bird on the cake']:
        assert classify({'bird'},caps+[text],{'bird'}) is None
    assert classify({'person'},['A man with a cockatiel'],{'person'}) is None
    assert classify({'person'},caps,{'person'}) is None
    assert classify({'person'},['A runner on the street'],{'person'})=='people'


def test_image_only_proxy():
    from scripts.bird_extremes import features
    a=features(Image.new('RGB',(256,256),'red'))
    b=features(Image.new('RGB',(256,256),'blue'))
    assert a['warm']>b['warm'] and a['strong']==b['strong']==0
    assert set(a)==set('strong bright saturated warm detailed contrasty busy'.split())
