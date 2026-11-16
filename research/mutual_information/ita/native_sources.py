#!/usr/bin/env python3
"""Read-only native-mask resolution, with exact isolated recovery of old PNGs.

The parent validates the TEST catalog, full-F ITA and scientific contracts.
This helper never regenerates a continuation case or changes an old artifact.
Only original_readonly cases lacking usable native files may call the frozen
batch.process_case, inside a new private directory. Adoption requires exact
PNG, decoded-array and count parity against the original compact receipt.
The controller uses only the standard library; numerical verification runs
under the frozen mask runtime. Production recovery uses CPU60-63, nice19,
no GPU and one case at a time (the frozen mask process uses two CV threads).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

SCHEMA = 'ita_stratified_native_sources_v1'
DOMAINS = ('P', 'D', 'O', 'F')
MODS = ('rgb', 'uv', 'plv', 'plh')
CPUS = {60, 61, 62, 63}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    with Path(path).open(encoding='utf8') as stream:
        return json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def descriptor(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def bound(value):
    require(isinstance(value, dict) and isinstance(value.get('path'), str), 'Missing descriptor')
    path = Path(value['path'])
    require(path.is_absolute(), 'Native-source descriptors must be absolute')
    require(path.is_file() and sha(path) == value.get('sha256'), 'Bound file missing or changed: ' + str(path))
    return path.resolve()


def seal(value):
    return dict(value, record_sha256=digest(value))


def checked(value):
    require(value.get('record_sha256') == digest({k: v for k, v in value.items() if k != 'record_sha256'}),
            'Record seal differs')
    return value


def save(path, value):
    path, payload = Path(path), encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == payload, 'Refusing to replace frozen native-source artifact: ' + str(path))
        return
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            require(path.read_bytes() == payload, 'Concurrent native-source publication differs')
    finally:
        if temporary.exists():
            temporary.unlink()


def disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a != b and a not in b.parents and b not in a.parents


def array_evidence(compact):
    summary_desc = compact.get('evidence', {}).get('mask/summary.json')
    summary = read(bound(summary_desc))
    require(summary.get('status') == 'complete' and summary.get('counts') == compact['counts_native'] and
            summary.get('native_shape') == compact['native_shape'] and summary.get('partition_verified') is True,
            'Original native summary/count/shape differs')
    hashes = summary.get('array_sha256', {})
    require(set(hashes) == set(DOMAINS), 'Incomplete original native-array hashes')
    require({d: summary['artifact_sha256'][d + '.png'] for d in DOMAINS} == compact['native_mask_file_sha256'],
            'Original native PNG inventory differs')
    return hashes, summary_desc


def validate_files(masks, compact):
    """PNG-byte equality implies the same decoded arrays; retain sealed evidence."""
    require(isinstance(masks, dict) and set(masks) == set(DOMAINS), 'Native mask inventory differs')
    normalized = {}
    missing = []
    for domain in DOMAINS:
        item = masks[domain]
        require(item.get('sha256') == compact['native_mask_file_sha256'][domain], 'Native PNG expected hash differs: ' + domain)
        path = Path(item['path'])
        require(path.is_absolute(), 'Native masks require absolute paths')
        if not path.is_file():
            missing.append(domain)
            continue
        require(sha(path) == item['sha256'], 'Native PNG content changed: ' + str(path))
        normalized[domain] = descriptor(path)
    return None if missing else normalized


def verify_decoded(request_path, output_path):
    """Numerical subprocess entry point under the original mask environment."""
    import numpy as np
    from PIL import Image
    request = read(request_path)
    require(request.get('helper_sha256') == sha(__file__), 'Numerical verifier source differs')
    arrays, counts, hashes = {}, {}, {}
    for domain in DOMAINS:
        path = bound(request['masks'][domain])
        with Image.open(path) as image:
            require(image.mode == 'L', 'Native mask is not uint8 single-channel')
            array = np.asarray(image).copy()
        require(array.dtype == np.uint8 and list(array.shape) == request['native_shape'], 'Native decoded geometry/type differs')
        require(np.isin(np.unique(array), [0, 255]).all(), 'Frozen native mask is not 0/255')
        counts[domain] = int(np.count_nonzero(array))
        hashes[domain] = hashlib.sha256(array.tobytes()).hexdigest()
        arrays[domain] = array != 0
    occupancy = arrays['P'].astype(np.uint8) + arrays['D'] + arrays['O']
    require(not np.any(occupancy > 1) and np.array_equal(occupancy != 0, arrays['F']), 'Recovered native partition differs')
    require(counts == request['counts_native'] and hashes == request['native_array_sha256'], 'Recovered native array/count mismatch')
    result = seal({'schema': SCHEMA + '_decoded_verification', 'status': 'complete',
                   'request': descriptor(request_path), 'helper_sha256': sha(__file__),
                   'masks': request['masks'], 'counts_native': counts,
                   'native_array_sha256': hashes, 'partition_verified': True})
    save(output_path, result)
    return result


@contextmanager
def recovery_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'recovery.lock'
    try:
        stream = path.open('x', encoding='utf8')
    except FileExistsError:
        raise RuntimeError('Recovery lock exists; verify its exact process before removing it: ' + str(path))
    try:
        with stream:
            json.dump({'pid': os.getpid(), 'helper_sha256': sha(__file__)}, stream)
        yield
    finally:
        path.unlink()


class NativeResolver:
    def __init__(self, request, source_rows, output):
        self.request = request
        self.rows = source_rows
        self.output = Path(output).resolve()
        self.old = Path(request['old_mask_root']).resolve()
        self.parallel = Path(request['parallel_mask_root']).resolve()
        require(disjoint(self.output, self.old) and disjoint(self.output, self.parallel), 'Native recovery overlaps old/current mask output')
        for key in ('batch_module', 'frozen_config', 'mask_contract'):
            path = bound(request[key])
            require(path != self.output and self.output not in path.parents, 'Recovery contains a frozen input')
        for row in source_rows:
            for name in MODS:
                p = Path(row[name]['path']).resolve()
                require(p != self.output and self.output not in p.parents, 'Recovery contains source images')
        self.contract = read(bound(request['mask_contract']))
        self._batch = None
        self._settings = None
        self.table1 = {}
        if request.get('table1_manifest'):
            manifest_path = bound(request['table1_manifest'])
            manifest = read(manifest_path)
            require(manifest.get('schema') == 'mi_atlas_six_metrics_input_v1' and manifest.get('status') == 'frozen', 'Unsupported Table1 native manifest')
            for row in manifest['rows']:
                require(row['sample_id'] not in self.table1, 'Duplicate Table1 sample ID')
                copy = dict(row)
                for name in MODS + DOMAINS:
                    p = Path(copy[name]['path'])
                    copy[name] = dict(copy[name], path=str((p if p.is_absolute() else manifest_path.parent / p).resolve()))
                self.table1[row['sample_id']] = copy

    def _identity(self, index, compact, ref, receipt_descriptor):
        require(type(index) is int and 0 <= index < len(self.rows), 'Bad catalog index')
        row = self.rows[index]
        path = bound(receipt_descriptor)
        require(checked(read(path)) == compact, 'Compact receipt argument differs from sealed file')
        require(compact.get('status') == 'complete' and compact['index'] == index and
                compact['sample_id'] == row['sample_id'] and compact['subject_id'] == row['subject_id'] and
                compact['native_shape'] == row['native_shape'] and compact['sources'] == {m: row[m] for m in MODS} and
                compact['contract_sha256'] == digest(self.contract), 'Native resolution identity/contract differs')
        require(ref.get('origin') in ('original_readonly', 'continuation'), 'Unrecognized native origin')
        root = self.old if ref['origin'] == 'original_readonly' else self.parallel / 'generated'
        require(path == (root / 'cases' / ('case_%06d' % index) / 'receipt.json').resolve(), 'Receipt path is not the designated origin')
        if ref['origin'] == 'continuation':
            checked(ref)
            require(ref['receipt'] == receipt_descriptor and ref['index'] == index and ref['sample_id'] == row['sample_id'], 'Continuation reference differs')
        return row, path

    def _retention(self, retention_descriptor, compact, expected_receipt=None, expected_directory=None):
        retained = checked(read(bound(retention_descriptor)))
        require(retained.get('schema') == 'three_region_owned_native_retention_v1' and retained['status'] == 'complete' and
                retained['index'] == compact['index'] and retained['sample_id'] == compact['sample_id'], 'Retention identity differs')
        if expected_receipt is not None:
            require(retained['compact_receipt'] == expected_receipt, 'Retention compact binding differs')
        retained_compact = checked(read(bound(retained['compact_receipt'])))
        for field in ('sample_id', 'subject_id', 'native_shape', 'sources', 'contract_sha256',
                      'counts_native', 'native_mask_file_sha256', 'mask_config_sha256'):
            require(retained_compact[field] == compact[field], 'Retention/original parity differs: ' + field)
        native_directory = Path(retained['native_directory']).resolve()
        if expected_directory is not None:
            require(native_directory == Path(expected_directory).resolve(), 'Retained native directory differs')
        require(bound(retained['owner']) == native_directory / 'owner.json', 'Retention ownership path differs')
        arrays, _ = array_evidence(compact)
        require(retained['counts_native'] == compact['counts_native'] and retained['native_array_sha256'] == arrays and
                retained.get('cleanup_replaced_by_retention') is True and retained.get('original_outputs_touched') is False,
                'Retained native parity/storage evidence differs')
        for domain, item in retained['native_masks'].items():
            require(Path(item['path']).resolve() == native_directory / 'mask' / (domain + '.png'), 'Retained native path differs')
        return validate_files(retained['native_masks'], compact)

    def _settings_for_recovery(self):
        if self._settings is not None:
            return self._batch, self._settings
        require(sys.platform.startswith('linux') and hasattr(os, 'sched_getaffinity'), 'Production native recovery requires configured Linux CPU isolation')
        affinity = set(os.sched_getaffinity(0))
        require(affinity and affinity.issubset(CPUS) and len(affinity) <= 4 and os.getpriority(os.PRIO_PROCESS, 0) >= 19,
                'Recovery must run only on CPU60-63 at nice19')
        require(os.environ.get('CUDA_VISIBLE_DEVICES', '') == '', 'GPU must remain disabled')
        module_path = bound(self.request['batch_module'])
        spec = importlib.util.spec_from_file_location('_ita_original_batch_native_recovery', module_path)
        batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batch)
        old = batch.load_settings(bound(self.request['frozen_config']), project_root=module_path.parents[1])
        for key in ('stage_code_sha256', 'frozen_source_sha256', 'vendor_sha256', 'model_sha256', 'parity_report_sha256',
                    'runtimes', 'controller_sha256', 'config_sha256', 'cohort_sha256'):
            require(self.contract[key] == old['contract'][key], 'Recovery scientific contract differs: ' + key)
        require(self.contract['schema'] == 'uv_three_region_test_mask_contract_v1' and
                self.contract['normalized_source_rows_sha256'] == digest(self.rows) and
                self.contract['sample_order'] == [r['sample_id'] for r in self.rows] and
                self.contract['test_sample_count'] == len(self.rows), 'Recovery TEST order differs')
        private = self.output / 'private_native'
        require(disjoint(private, old['output']), 'Recovery overlaps original batch')
        self._batch = batch
        self._settings = dict(old, rows=self.rows, output=private, contract=self.contract, workers=1, pilot=1)
        return batch, self._settings

    def _recover(self, index, compact, receipt_descriptor):
        require(self.request.get('allow_old_native_recovery') is True, 'Old native recovery is not authorized')
        batch, settings = self._settings_for_recovery()
        root = self.output / 'private_native'
        case = root / 'cases' / ('case_%06d' % index)
        require(case.resolve().parent == (root / 'cases').resolve() and self.output in case.resolve().parents,
                'Recovery target escaped owned root')
        for p in (self.output, root, root / 'cases', case):
            require(not p.is_symlink(), 'Symlinked recovery target')
        owner = seal({'schema': SCHEMA + '_owner', 'index': index, 'sample_id': compact['sample_id'],
                      'request_sha256': digest(self.request), 'compact_receipt': receipt_descriptor,
                      'mask_contract': self.request['mask_contract'], 'helper_sha256': sha(__file__),
                      'source_row_sha256': digest(self.rows[index])})
        case.mkdir(parents=True, exist_ok=True)
        save(case / 'owner.json', owner)
        with recovery_lock(root):
            journal = batch.Journal(settings)
            original = batch.process_case(settings, index, journal)
            require(original['status'] == 'complete' and original['index'] == index and
                    original['sample_id'] == compact['sample_id'] and original['subject_id'] == compact['subject_id'] and
                    original['native_shape'] == compact['native_shape'] and original['sources'] == compact['sources'] and
                    original['contract_sha256'] == compact['contract_sha256'] and original['counts'] == compact['counts_native'] and
                    original['mask_config_sha256'] == compact['mask_config_sha256'], 'Recovered case metadata/count/config differs')
            masks = validate_files(original['masks'], compact)
            require(masks is not None, 'Recovered native files missing')
            arrays, _ = array_evidence(compact)
            verify_request = {'schema': SCHEMA + '_decoded_request', 'helper_sha256': sha(__file__), 'masks': masks,
                              'native_shape': compact['native_shape'], 'counts_native': compact['counts_native'],
                              'native_array_sha256': arrays, 'compact_receipt': receipt_descriptor}
            request_path = case / 'decoded_request.json'
            proof_path = case / 'decoded_verification.json'
            save(request_path, verify_request)
            runtime = settings['runtimes']['mask']
            if proof_path.exists():
                proof = checked(read(proof_path))
                require(proof['request'] == descriptor(request_path) and proof['masks'] == masks and
                        proof['native_array_sha256'] == arrays and proof['counts_native'] == compact['counts_native'] and
                        proof['partition_verified'] is True, 'Existing decoded proof differs')
            else:
                command = [runtime['python'], str(Path(__file__).resolve()), '--verify-request', str(request_path),
                           '--verify-output', str(proof_path)]
                log = case / ('decoded_' + uuid.uuid4().hex + '.log')
                with log.open('xb') as stream:
                    result = subprocess.run(command, env=batch.child_env(runtime), stdin=subprocess.DEVNULL,
                                            stdout=stream, stderr=subprocess.STDOUT, check=False)
                require(result.returncode == 0, 'Decoded native parity check failed; see ' + str(log))
                proof = checked(read(proof_path))
                require(proof['request'] == descriptor(request_path) and proof['counts_native'] == compact['counts_native'] and
                        proof['native_array_sha256'] == arrays, 'Decoded native subprocess proof differs')
        return masks, {'recovered_case_receipt': descriptor(case / 'case_receipt.json'),
                       'owner': descriptor(case / 'owner.json'), 'decoded_verification': descriptor(proof_path)}

    def resolve(self, index, compact, ref, receipt_descriptor=None):
        require(receipt_descriptor is not None, 'Explicit sealed compact receipt descriptor required')
        row, _ = self._identity(index, compact, ref, receipt_descriptor)
        arrays, summary_descriptor = array_evidence(compact)
        proof_path = self.output / 'sources' / ('%06d.json' % index)
        if proof_path.exists():
            proof = checked(read(proof_path))
            require(proof['helper_sha256'] == sha(__file__) and proof['request_sha256'] == digest(self.request) and
                    proof['compact_receipt'] == receipt_descriptor and proof['sample_id'] == row['sample_id'] and
                    proof['native_array_sha256'] == arrays and proof['counts_native'] == compact['counts_native'], 'Cached native proof binding differs')
            masks = validate_files(proof['masks'], compact)
            require(masks is not None, 'Previously frozen native source disappeared')
            for value in proof.get('supporting_evidence', {}).values():
                bound(value)
            return {'status': 'ready', 'masks': masks, 'proof': descriptor(proof_path), 'reason': None}

        masks, mode, supporting = None, None, {}
        if ref['origin'] == 'continuation':
            retention_descriptor = ref.get('native_retention')
            expected = self.parallel / 'generated/private_native/cases' / ('case_%06d' % index)
            masks = self._retention(retention_descriptor, compact, receipt_descriptor, expected)
            require(masks is not None, 'Published continuation native source disappeared; do not regenerate')
            mode, supporting = 'continuation_retained', {'native_retention': retention_descriptor}
        else:
            # 优先使用原件
            native_case_descriptor = compact.get('evidence', {}).get('case_receipt.json')
            if native_case_descriptor:
                native_case = checked(read(bound(native_case_descriptor)))
                require(native_case['sample_id'] == row['sample_id'] and native_case['sources'] == compact['sources'] and
                        native_case['counts'] == compact['counts_native'], 'Archived native case binding differs')
                masks = validate_files(native_case['masks'], compact)
                if masks:
                    mode, supporting = 'original_native_existing', {'native_case': native_case_descriptor}
            if masks is None and row['sample_id'] in self.table1:
                candidate = self.table1[row['sample_id']]
                require(candidate['subject_id'] == row['subject_id'] and candidate['native_shape'] == row['native_shape'] and
                        all(candidate[m] == row[m] for m in MODS), 'Table1 candidate source identity differs')
                masks = validate_files({d: candidate[d] for d in DOMAINS}, compact)
                if masks:
                    mode, supporting = 'table1_native_existing', {'table1_manifest': self.request['table1_manifest']}
            if masks is None:
                pilot = self.parallel / 'pilot/cases' / ('case_%06d' % index) / 'retention.json'
                if pilot.exists():
                    masks = self._retention(descriptor(pilot), compact,
                        expected_directory=self.parallel / 'pilot/private_native/cases' / ('case_%06d' % index))
                    if masks:
                        mode, supporting = 'pilot_native_existing', {'pilot_retention': descriptor(pilot)}
            if masks is None:
                if self.request.get('allow_old_native_recovery') is not True:
                    return {'status': 'pending', 'masks': None, 'proof': None, 'reason': 'old_native_files_unavailable'}
                masks, supporting = self._recover(index, compact, receipt_descriptor)
                mode = 'old_native_exact_recovery'

        require(masks is not None and mode is not None, 'Native resolver failed to bind source')
        proof = seal({'schema': SCHEMA, 'status': 'ready', 'index': index, 'sample_id': row['sample_id'],
                      'subject_id': row['subject_id'], 'source_row_sha256': digest(row),
                      'helper_sha256': sha(__file__), 'request_sha256': digest(self.request),
                      'compact_receipt': receipt_descriptor, 'mask_contract': self.request['mask_contract'],
                      'mode': mode, 'masks': masks, 'native_array_sha256': arrays,
                      'counts_native': compact['counts_native'], 'native_shape': compact['native_shape'],
                      'original_summary': summary_descriptor, 'supporting_evidence': supporting,
                      'original_outputs_modified': False, 'continuation_regenerated': False})
        save(proof_path, proof)
        return {'status': 'ready', 'masks': masks, 'proof': descriptor(proof_path), 'reason': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-request', type=Path, required=True)
    parser.add_argument('--verify-output', type=Path, required=True)
    args = parser.parse_args()
    verify_decoded(args.verify_request, args.verify_output)


if __name__ == '__main__':
    main()
