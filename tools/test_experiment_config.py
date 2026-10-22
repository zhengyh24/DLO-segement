import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiment_config import apply_config

class ConfigTests(unittest.TestCase):
    def run_config(self,values,cli=()):
        p=argparse.ArgumentParser()
        p.add_argument('--experiment-config',type=Path)
        p.add_argument('--crop-size',type=int,default=512)
        p.add_argument('--preserve-aspect',action='store_true')
        p.add_argument('--output',type=Path)
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'test.json'
            config.write_text(json.dumps({'stage_a':values}))
            argv=['--experiment-config',str(config),*cli]
            with patch.object(sys,'argv',['test',*argv]):
                return apply_config(p,p.parse_args(argv),'stage_a')

    def test_explicit_cli_wins(self):
        self.assertEqual(self.run_config({'crop_size':1024},['--crop-size','768']).crop_size,768)
        self.assertEqual(self.run_config({'crop_size':1024}).crop_size,1024)

    def test_no_io_override(self):
        with self.assertRaises(ValueError):self.run_config({'output':'unsafe'})

    def test_boolean_is_not_string(self):
        with self.assertRaises(ValueError):self.run_config({'preserve_aspect':'false'})

if __name__=='__main__':unittest.main()
