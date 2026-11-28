import tempfile
import unittest
from pathlib import Path

from resume_models import check_group, preserve_partial


class ResumeTests(unittest.TestCase):
    def test_incomplete_group_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            group = run / 'predictions/test/L12'
            group.mkdir(parents=True)
            (group / 'diagnostics.json').write_text('{}')
            self.assertFalse(check_group(run, 'test', 12, {'daily_quantiles': [.5], 'period_quantiles': [.5]}))

    def test_partial_results_preserved_and_other_groups_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            for kind in ['models', 'predictions']:
                for level in [11, 12]:
                    path = run / kind / 'test' / f'L{level:02d}'
                    path.mkdir(parents=True)
                    (path / 'saved.txt').write_text('keep')
            preserve_partial(run, 'test', 12)
            self.assertTrue((run / 'models/test/L11/saved.txt').exists())
            self.assertFalse((run / 'models/test/L12').exists())
            backups = list((run.parent / 'run_interrupted').rglob('saved.txt'))
            self.assertEqual(len(backups), 2)
            self.assertTrue(all(p.read_text() == 'keep' for p in backups))


if __name__ == '__main__':
    unittest.main()
