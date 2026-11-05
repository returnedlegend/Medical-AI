"""Research bundles preserve the protocol and distinguish incomplete runs."""
import json
from unittest.mock import patch

import pytest

from retraction.study import run_study


def test_complete_bundle_references_real_run_and_refuses_overwrite(tmp_path):
    output = tmp_path / 'study'
    complete = run_study(output, [{'n_patients': 40, 'seed': 42}])
    protocol = json.loads((output / 'protocol.json').read_text())
    assert complete['protocol_sha256'] == protocol['protocol_sha256']
    assert complete['completed_runs'] == 1
    payload = json.loads((output / complete['files'][0]).read_text())
    assert payload['results']['summary']['n_patients'] == 40
    assert payload['results']['summary']['test_size'] == 12
    assert (output / 'summary.csv').read_text().count('\n') == 2
    assert 'not a time-stamped external preregistration' in (output / 'report.md').read_text()
    original = (output / 'protocol.json').read_bytes()
    with pytest.raises(FileExistsError):
        run_study(output, [{'n_patients': 40, 'seed': 43}])
    assert (output / 'protocol.json').read_bytes() == original


def test_failure_keeps_protocol_without_completion_claim(tmp_path):
    output = tmp_path / 'interrupted'
    with patch('retraction.study.run_experiment', side_effect=RuntimeError('numerical failure')):
        with pytest.raises(RuntimeError):
            run_study(output, [{'n_patients': 40}])
    assert (output / 'protocol.json').is_file()
    assert not (output / 'complete.json').exists()


def test_duplicate_seeds_are_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match='Duplicate'):
        run_study(tmp_path / 'duplicate', [{'n_patients': 40}, {'n_patients': 40}])
    assert not (tmp_path / 'duplicate').exists()
