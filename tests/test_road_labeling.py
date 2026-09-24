"""Checks that short scene histories cannot cross scene boundaries or lose targets."""
import tempfile
import unittest
from pathlib import Path

from scripts.label_road_with_qwen import parse_answer, sequence_for_target


class FakeNuScenes:
    def __init__(self, root):
        self.dataroot = str(root)
        self.samples = {}
        self.sample_data = {}
        for i in range(3):
            token = f'sample-{i}'
            image = root / f'frame-{i}.jpg'
            image.write_bytes(b'example')
            self.samples[token] = {
                'token': token, 'scene_token': 'one-scene',
                'prev': f'sample-{i-1}' if i else '',
                'data': {'CAM_FRONT': f'image-{i}'},
            }
            self.sample_data[f'image-{i}'] = {'filename': image.name}

    def get(self, table, token):
        return (self.samples if table == 'sample' else self.sample_data)[token]


class RoadLabelingTest(unittest.TestCase):
    def test_short_history_padding_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            nusc = FakeNuScenes(Path(directory))
            first = {'sample_token': 'sample-0', 'scene_token': 'one-scene'}
            third = {'sample_token': 'sample-2', 'scene_token': 'one-scene'}
            with self.assertRaises(ValueError):
                sequence_for_target(nusc, first, 4, 1, 'CAM_FRONT')
            self.assertEqual(
                [Path(p).name for p in sequence_for_target(nusc, first, 4, 1, 'CAM_FRONT', True)],
                ['frame-0.jpg'] * 4,
            )
            self.assertEqual(
                [Path(p).name for p in sequence_for_target(nusc, third, 4, 1, 'CAM_FRONT', True)],
                ['frame-0.jpg', 'frame-0.jpg', 'frame-1.jpg', 'frame-2.jpg'],
            )
            with self.assertRaises(ValueError):
                sequence_for_target(nusc, {'sample_token': 'sample-2', 'scene_token': 'other'},
                                    4, 1, 'CAM_FRONT', True)

    def test_response_rejects_invalid_class(self):
        self.assertIsNone(parse_answer('{"proposed_label":null,"confidence":"low"}')['proposed_label'])
        with self.assertRaises(ValueError):
            parse_answer('{"proposed_label":5,"confidence":"high"}')


if __name__ == '__main__':
    unittest.main()
