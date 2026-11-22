"""Two selective, cue-grounded impressions crossed with each training invitation."""
import argparse
from collections import Counter
import q38_quality020 as prior

parent = prior.parent
HERE = prior.HERE.parent/'025'
TARGETS = {
    '000000068293': ('The throat flashes red against a brilliant green head; that splash of color is lovely.', 'The bird holds so still that its narrow beak could pass for a twig.'),
    '000000117789': ('That dripping face and lolling tongue make the dog look wonderfully goofy.', 'There is so much open water around this wet dog; even the little wave leaves the frame feeling spacious.'),
    '000000137788': ('A fluffy dog tucked among the groceries is a funny thing to find on a pantry shelf.', 'The shelves are crammed, but the dog curled up below them looks utterly at rest.'),
    '000000171484': ('The giraffes seem rooted in place, their long legs planted apart.', 'Gray walls and brown spots give the enclosure a muted palette.'),
    '000000177529': ('That tiny kitten reaching toward the enormous muzzle makes an endearing pair.', 'Their noses nearly touch, but the small gap makes the encounter feel gentle rather than threatening.'),
    '000000221542': ('Those outsized feet and that jutting beak make the bird look imposing rather than cuddly.', 'I like how the bird stands out even against the dull colors of dust and corrugated metal.'),
    '000000494973': ('The lit blue wing and white head look lovely against those nearly black leaves.', 'Nothing in the perched bird seems to stir, even on that sharply sloping branch.'),
    '000000516947': ('The broad pale ripples leave the heron a lot of open space.', 'Its outstretched neck looks intent, not playful; the grays of the bird and water are subdued.'),
}


def recipe():
    rows, meta = prior.checked()
    seen = Counter()
    result = []
    for n,k,p,_ in rows:
        targets = TARGETS[n.removeprefix('images/').removesuffix('.jpg')]
        result.append((n,k,p,targets[seen[n,p] % 2]))
        seen[n,p] += 1
    return result, dict(meta, source_schedule_sha256=meta['schedule_sha256'], schedule_sha256=prior.fit.checksum(result))


def checked():
    items, meta = recipe()
    if meta['schedule_sha256'] != (HERE/'schedule.sha256').read_text().strip():
        raise ValueError('schedule changed')
    return items,meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check','train','before','after'))
    stage = parser.parse_args().stage
    if stage == 'train':
        items, meta = checked()
        earlier = set(meta['slot_substitution'].values())
        prior.fit.BANK = prior.MixedBank({n:(prior.OLD_BANK if n in earlier else prior.prev.fresh.BANK)/n for n,*_ in items})
    parent.checked = checked
    parent.dev.checked = prior.dev_panel
    parent.OUT = parent.OUT.parent/'overnight-025'
    parent.main(stage)
