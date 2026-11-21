import json
import tempfile
import unittest
from pathlib import Path


class DataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.axes: dict = {'custom': {'low': 'cool', 'mid': 'mild', 'high': 'warm'},
                     'other': {'low': 'a', 'mid': 'b', 'high': 'c'}}
        self.write_axes()
        self.rows = []
        for bank in ('liked', 'control'):
            (self.root / bank).mkdir()
            for i in range(16):
                image = f'{bank}/{i}.ppm'
                (self.root / image).write_bytes(b'P6\n1 1\n255\n' + bytes([i, bank == 'liked', 0]))
                self.rows.append({'image': image, 'group': f'scene-{i}',
                                  'scores': {'custom': (i % 4) / 3, 'other': None}})
        self.write_rows()

    def write_rows(self):
        (self.root / 'ratings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))

    def write_axes(self):
        (self.root / 'axes.yaml').write_text(json.dumps(self.axes))  # JSON is valid YAML

    def prepare(self, **options):
        from second_look.data import prepare
        return prepare(self.root, 'custom', **options)

    def test_prepares_seeded_group_disjoint_bounded_pairs(self):
        result = self.prepare(limit=5)
        self.assertEqual(set(result), {'train', 'test', 'singles'})
        self.assertEqual(result, self.prepare(limit=5))
        self.assertNotEqual(result, self.prepare(limit=5, seed=999))
        scores = {r['image']: r['scores']['custom'] for r in self.rows}
        heldout = {s['image'] for s in result['singles']}
        self.assertEqual({s['bank'] for s in result['singles']}, {'liked', 'control'})
        self.assertTrue(all(s == {'image': s['image'], 'bank': s['image'].split('/')[0]}
                            for s in result['singles']))
        self.assertEqual(len(heldout), len(result['singles']))
        for r in self.rows:
            self.assertEqual(r['image'] in heldout, f"control/{r['image'].split('/')[1]}" in heldout)
        for split in ('train', 'test'):
            pairs = result[split]
            self.assertTrue(0 < len(pairs) <= 5)
            self.assertEqual(len({frozenset(p['images']) for p in pairs}), len(pairs))
            self.assertLessEqual(abs(sum(1 if p['answer'] == 'A' else -1 for p in pairs)), 1)
            for p in pairs:
                self.assertEqual(set(p), {'images', 'prompt', 'answer'})
                a, b = p['images']
                self.assertEqual(a in heldout, split == 'test')
                self.assertEqual(b in heldout, split == 'test')
                self.assertGreaterEqual(abs(scores[a] - scores[b]), .15)
                self.assertEqual(p['answer'], 'A' if scores[a] > scores[b] else 'B')
                self.assertTrue(p['prompt'].strip())

    def test_low_direction_custom_question_and_nulls(self):
        self.axes['custom'].update(direction='low', question='Which feels cooler?')
        self.write_axes()
        for r in self.rows[::5]:
            r['scores']['custom'] = None
        self.write_rows()
        scores = {r['image']: r['scores']['custom'] for r in self.rows}
        data = self.prepare(gap=0)
        for p in data['train'] + data['test']:
            self.assertIn('Which feels cooler?', p['prompt'])
            a, b = (scores[x] for x in p['images'])
            self.assertNotEqual(a, b)
            self.assertEqual(p['answer'], 'A' if a < b else 'B')

    def test_validates_even_unselected_scores_and_groups(self):
        original = self.rows[0]
        for fields in ({'scores': {'custom': 0}}, {'scores': {'custom': 0, 'other': True}},
                       {'scores': {'custom': 0, 'other': 1.1}}, {'scores': {'custom': 0, 'other': float('nan')}},
                       {'group': 3}, {'group': ''}):
            with self.subTest(fields=fields):
                self.rows[0] = {**original, **fields}
                self.write_rows()
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_validates_axis_definitions_and_options(self):
        for attrs in ({'direction': 'sideways'}, {'mid': None}, {'question': ''}):
            with self.subTest(attrs=attrs):
                self.axes['other'] = dict(self.axes['custom'], **attrs)
                self.write_axes()
                with self.assertRaises(ValueError):
                    self.prepare()
        self.axes['other'] = self.axes['custom'].copy()
        self.write_axes()
        for options in ({'gap': -1}, {'gap': float('nan')}, {'limit': -1}, {'limit': 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)

    def test_rejects_missing_unsafe_and_duplicate_images(self):
        original = self.rows[0]['image']
        link = self.root / 'liked/symlink.ppm'
        link.symlink_to(self.root / 'control/0.ppm')
        for image in ('liked/missing.ppm', '/etc/hosts', 'liked/../control/0.ppm',
                      'unknown/x.ppm', 'liked/symlink.ppm', self.rows[1]['image']):
            with self.subTest(image=image):
                self.rows[0]['image'] = image
                self.write_rows()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.prepare()
        link.unlink()
        self.rows[0]['image'] = original
        self.write_rows()
        image = self.root / original
        image.write_bytes((self.root / 'control/2.ppm').read_bytes())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.prepare()
        image.write_bytes(b'not an image')
        with self.assertRaisesRegex(ValueError, 'image'):
            self.prepare()

    def test_rejects_duplicate_after_exif_rotation_and_rgb_conversion(self):
        from PIL import Image

        source = Image.new('RGBA', (2, 1), (255, 0, 0, 32))
        source.putpixel((1, 0), (0, 0, 255, 64))
        exif = Image.Exif()
        exif[274] = 6  # Display rotated 90 degrees clockwise.
        source.save(self.root / 'liked/tagged.png', exif=exif)
        source.transpose(Image.Transpose.ROTATE_270).convert('RGB').save(self.root / 'control/upright.png')
        for path in ('liked/tagged.png', 'control/upright.png'):
            self.rows.append(dict(self.rows[0], image=path))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'duplicate image'):
            self.prepare()

    def test_small_ungrouped_banks_have_usable_pairs_in_both_splits(self):
        keep = [self.rows[i] for i in (0, 4, 17, 21)]
        for r in self.rows:
            if r not in keep:
                (self.root / r['image']).unlink()
        self.rows = keep
        for r in self.rows:
            r.pop('group')
        self.write_rows()
        for seed in range(12):
            data = self.prepare(seed=seed)
            for split in ('test', 'train'):
                self.assertEqual(len(data[split]), 1)
                self.assertEqual({x.split('/')[0] for x in data[split][0]['images']}, {'liked', 'control'})

    def test_rejects_impossible_groups_or_uninformative_scores(self):
        for r in self.rows:
            r['group'] = r['image'].split('/')[0]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'bank|group'):
            self.prepare()
        for r in self.rows:
            r.pop('group')
            r['scores']['custom'] = 0.5
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'pair|gap'):
            self.prepare(gap=0)

    def test_signed_scale_allows_full_span_gap(self):
        for r in self.rows:
            r['scores']['custom'] = 1 if r['image'].startswith('liked/') else -1
        self.write_rows()
        result = self.prepare(gap=2)
        for split in ('train', 'test'):
            self.assertTrue(result[split])
            for p in result[split]:
                self.assertTrue(p['images']['AB'.index(p['answer'])].startswith('liked/'))

    def test_every_bank_photo_needs_a_rating_row(self):
        self.rows.pop()
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'unrated|missing rating'):
            self.prepare()
