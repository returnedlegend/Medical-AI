"""Validate the actual headless command entry points and their output contract."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def invoke(module, *args):
    return subprocess.run([sys.executable, '-m', module, *map(str, args)],
                          cwd=Path(__file__).resolve().parents[1],
                          text=True, capture_output=True, timeout=30)


def test_single_cli_executes_and_refuses_overwrite(tmp_path):
    output = tmp_path / 'result.json'
    run = invoke('retraction', '--patients', 40, '--seed', 8, '--output', output)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text())
    assert result['results']['summary']['n_patients'] == 40
    assert result['results']['config']['seed'] == 8
    original = output.read_bytes()
    refused = invoke('retraction', '--patients', 40, '--output', output)
    assert refused.returncode == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize('module', ['retraction', 'retraction.study'])
def test_invalid_config_type_is_actionable(module, tmp_path):
    config = tmp_path / 'bad.json'
    config.write_text('[]')
    response = invoke(module, '--config', config, '--patients', 40, '--output', tmp_path / 'out')
    assert response.returncode == 2
    assert 'Configuration JSON must be an object' in response.stderr
    assert 'Traceback' not in response.stderr
    assert not (tmp_path / 'out').exists()


def test_study_cli_executes_requested_grid(tmp_path):
    output = tmp_path / 'bundle'
    run = invoke('retraction.study', '--seeds', 10, 11, '--profiles', 'nominal',
                 '--patients', 40, '--output', output)
    assert run.returncode == 0, run.stderr
    complete = json.loads((output / 'complete.json').read_text())
    assert complete['completed_runs'] == 2
    protocol = json.loads((output / 'protocol.json').read_text())['protocol']
    assert [cfg['seed'] for cfg in protocol['configs']] == [10, 11]
    assert {cfg['threshold_profile'] for cfg in protocol['configs']} == {'nominal'}
