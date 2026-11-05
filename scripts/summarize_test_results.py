#!/usr/bin/env python3
"""Expose real pytest JUnit failures in the public GitHub job summary."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('test-results/backend.xml'))
    args = parser.parse_args()
    if not args.report.is_file():
        text = '# Backend test evidence\n\nNo JUnit report was produced. Tests may not have started; this is not a passing result.\n'
    else:
        root = ET.parse(args.report).getroot()
        suites = [root] if root.tag == 'testsuite' else list(root.findall('testsuite'))
        counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        text = '# Backend test evidence\n\n' + ', '.join(f'{key}: {value}' for key, value in counts.items()) + '\n\n'
        for case in root.iter('testcase'):
            for kind in ('failure', 'error'):
                detail = case.find(kind)
                if detail is not None:
                    label = f'{case.get("classname", "")}.{case.get("name", "")}'
                    excerpt = (detail.text or detail.get('message', ''))[-12000:].replace('```', "'''")
                    text += f'## {label}\n\n```text\n{excerpt}\n```\n\n'
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with Path(summary).open('a') as out:
            out.write(text)
    print(text)
    # The original pytest step owns its exit status. Reporting is read-only;
    # failed tests remain failed and are never replaced by this helper's result.


if __name__ == '__main__':
    main()
