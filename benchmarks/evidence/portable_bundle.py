"""Archive merged WireSeg evidence, following the declared run/source directories.

make-spec freezes the two source versions and their 39+3 training membership.
build reads terminal artifacts only. verify checks a downloaded zip without needing
the training host, data, torch, or the source directories. Python stdlib only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


ARCHIVE = 'pilot_evidence_results.zip'
MANIFEST = 'download_manifest.json'
RECEIPT = 'package_receipt.json'
CHUNK = 1024 * 1024
EXTENSIONS = set('.json .jsonl .csv .tsv .png .jpg .jpeg .webp .pdf .log .out .err .md .html .txt .npz .py .sh .patch .diff .yaml .yml .toml .svg'.split())
SKIP_DIRS = {'.git', '__pycache__', '.pytest_cache', '.venv', '.portable_bundle.stage'}
GENERATED = {ARCHIVE, MANIFEST, RECEIPT, '.portable_bundle.lock'}


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def json_bytes(data):
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode('utf-8')


def digest_stream(stream, destination=None):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK):
        if destination is not None:
            destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return {'sha256':digest.hexdigest(), 'bytes':size}


def digest_file(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def archive_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or name != path.as_posix() or not path.parts:
        raise ValueError(f'Unsafe or ambiguous archive member: {name}')
    return name


def tag(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Invalid evidence label: {value}')
    return value


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.part')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def output_lock(output):
    """OS-owned lock is automatically released after supervisor termination."""
    with (output/'.portable_bundle.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if not handle.tell():
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def evidence_files(path, allowed, skip=None, explicit=False):
    """Follow named directory links, reject broken/escaping links and ancestry cycles."""
    path = Path(path)
    skip = set(skip or ())

    def walk(current, relative, ancestors):
        resolved = current.resolve(strict=True)
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise ValueError(f'Link target outside declared input roots: {current} -> {resolved}')
        if resolved in skip:
            return
        mode = resolved.stat().st_mode
        if stat.S_ISDIR(mode):
            if resolved in ancestors:
                raise ValueError(f'Directory link cycle: {current} -> {resolved}')
            for child in sorted(current.iterdir(), key=lambda p:p.name):
                if child.name in SKIP_DIRS or child.name in GENERATED or child.name.startswith('.portable_bundle.'):
                    continue
                yield from walk(child, relative / child.name, ancestors | {resolved})
        elif stat.S_ISREG(mode):
            if current.suffix.lower() in EXTENSIONS:
                yield relative.as_posix(), resolved
            elif explicit and current == path:
                raise ValueError(f'Unsupported evidence file type: {current}')

    relative = PurePosixPath(path.name) if path.is_file() else PurePosixPath()
    yield from walk(path, relative, set())


def make_spec(args):
    merged = Path(args.merged_runs).resolve(strict=True)
    spec = dict(schema=1, merged_runs=str(merged), report=str(merged/'final_results'), versions={}, inputs=[], evidence={})
    for version, selected_stage, count in (('original', False, 39), ('recovery', True, 3)):
        source, protocol, runs = [Path(getattr(args, version+'_'+name)).resolve(strict=True) for name in ('source','protocol','runs')]
        frozen = read_json(protocol)
        jobs = [j['id'] for j in frozen['jobs'] if (j['stage']=='native80') == selected_stage]
        if len(set(jobs)) != count or len(jobs) != count:
            raise ValueError(f'{protocol}: expected {count} distinct {version} jobs, found {len(jobs)}')
        spec['versions'][version] = dict(source=str(source), protocol=str(protocol), runs=str(runs), jobs=jobs)
    for item in args.evidence:
        label, sep, value = item.partition('=')
        if not sep or label in spec['evidence']:
            raise ValueError(f'Duplicate or malformed evidence: {item}')
        spec['evidence'][tag(label)] = str(Path(value).resolve(strict=True))
    spec['inputs'] = [str(Path(p).resolve(strict=True)) for p in args.input]
    path = Path(args.write)
    if path.exists() and read_json(path) != spec:
        raise FileExistsError(f'Different frozen spec already exists: {path}')
    atomic_bytes(path, json_bytes(spec))
    return spec


def validate_source_zip(path):
    """An explicitly supplied source transport must contain source files only."""
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names) != len(set(names)) or not any(n.endswith('.py') for n in names):
            raise ValueError(f'Empty/duplicate source archive: {path}')
        for name in names:
            archive_name(name)
            if Path(name).suffix.lower() not in {'.py','.md','.json','.sh','.txt','.yaml','.yml','.toml'}:
                raise ValueError(f'Unexpected member in source transport: {name}')


class Inventory:
    def __init__(self, spec):
        self.spec = spec
        self.paths = {}
        self.history = {}
        self.versions = {}
        self.protocol_inputs = {}

    def add(self, name, path):
        name = archive_name(name)
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f'Expected a file: {path}')
        if name in self.paths and self.paths[name] != resolved:
            raise ValueError(f'Two files map to {name}')
        self.paths[name] = resolved

    def add_tree(self, prefix, folder, allowed, skip=None):
        for relative, path in evidence_files(folder, allowed, skip, explicit=True):
            self.add(prefix+'/'+relative, path)

    def protocol_input(self, path):
        path = Path(path).resolve(strict=True)
        if path not in self.protocol_inputs:
            fingerprint = digest_file(path)['sha256']
            name = 'protocol_inputs/'+fingerprint+'_'+path.name
            self.add(name, path)
            self.protocol_inputs[path] = name

    def collect(self, output):
        spec = self.spec
        merged = Path(spec['merged_runs']).resolve(strict=True)
        report = Path(spec['report']).resolve(strict=True)
        versions = spec['versions']
        version_runs = [Path(v['runs']).resolve(strict=True) for v in versions.values()]
        for path in [merged/'completion.json', merged/'expected_evaluations.json', merged/'task_receipts.json', report/'analysis.json', report/'RESULTS_ZH.html']:
            if not path.is_file():
                raise FileNotFoundError(f'Terminal evidence missing: {path}')
        allowed = {merged, *version_runs}
        self.add_tree('runs', merged, allowed, skip={report, output} - {merged})
        self.add_tree('runs/final_results', report, {report}, skip={output} - {report})
        seen_jobs = set()
        for label, item in versions.items():
            tag(label)
            selected = set(item['jobs'])
            if seen_jobs & selected or len(selected) != len(item['jobs']):
                raise ValueError(f'Duplicate selected training membership for {label}')
            seen_jobs.update(selected)
            source = Path(item['source']).resolve(strict=True)
            if source.is_file() and source.suffix.lower() == '.zip':
                validate_source_zip(source)
                self.add(f'versions/{label}/source/'+source.name, source)
            else:
                self.add_tree(f'versions/{label}/source', source, {source})
            if not any(n.startswith(f'versions/{label}/source/') for n in self.paths):
                raise ValueError(f'No engine source artifacts for {label}')
            self.add(f'versions/{label}/protocol.json', item['protocol'])
            protocol = read_json(item['protocol'])
            jobs = {job['id']:job for job in protocol['jobs']}
            if not selected <= jobs.keys():
                raise ValueError(f'Unknown selected jobs in {label}')
            for job in jobs.values():
                self.add(f'versions/{label}/configs/'+tag(job['id'])+'.json', job['config_path'])
                self.protocol_input(job['config']['manifest'])
            self.protocol_input(protocol['test_manifest'])
            for task in protocol.get('additional_evaluations', []):
                for path in task.get('inputs', []):
                    self.protocol_input(path)
            runs = Path(item['runs']).resolve(strict=True)
            self.versions[label] = dict(selected_jobs=sorted(selected), protocol_member=f'versions/{label}/protocol.json',
                                        history_prefix=f'history/{label}/')
            for relative, path in evidence_files(runs, {runs, *version_runs}, skip={output}):
                self.history['history/'+label+'/'+relative] = (path, 'runs/'+relative)
        for path in spec.get('inputs', []):
            self.protocol_input(path)
        for label, value in spec.get('evidence', {}).items():
            path = Path(value).resolve(strict=True)
            self.add_tree('evidence/'+tag(label), path, {path}, skip={output}-{path})
        return self


def write_file(bundle, name, path):
    # Precompressed media and numeric archives do not benefit from another DEFLATE pass.
    info = zipfile.ZipInfo(archive_name(name))
    info.compress_type = zipfile.ZIP_STORED if Path(name).suffix.lower() in {'.png','.npz','.jpg','.jpeg','.pdf','.webp'} else zipfile.ZIP_DEFLATED
    with path.open('rb') as source, bundle.open(info, 'w', force_zip64=True) as dest:
        before = os.fstat(source.fileno())
        result = digest_stream(source, dest)
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or result['bytes'] != after.st_size:
        raise RuntimeError(f'Input changed while archiving: {path}')
    return result


def archive_inventory(archive, inventory, spec_hash):
    members, aliases, divergent = {}, {}, []
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as bundle:
        for name, path in sorted(inventory.paths.items()):
            members[name] = write_file(bundle, name, path)
        for name, (path, equivalent) in sorted(inventory.history.items()):
            current = inventory.paths.get(equivalent)
            if current is not None and (path == current or digest_file(path) == members[equivalent]):
                aliases[name] = equivalent
            else:
                members[name] = write_file(bundle, name, path)
                if current is not None:
                    divergent.append(dict(parent_member=name, merged_member=equivalent))
        redirect = b'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../runs/final_results/RESULTS_ZH.html"><a href="../runs/final_results/RESULTS_ZH.html">WireSeg results</a>'
        bundle.writestr('report/RESULTS_ZH.html', redirect)
        members['report/RESULTS_ZH.html'] = dict(sha256=hashlib.sha256(redirect).hexdigest(), bytes=len(redirect))
        spec_bytes = json_bytes(inventory.spec)
        bundle.writestr('provenance/bundle_spec.json', spec_bytes)
        members['provenance/bundle_spec.json'] = dict(sha256=hashlib.sha256(spec_bytes).hexdigest(), bytes=len(spec_bytes))
        missing = {}
        origin_problems = {}
        for label, version in inventory.versions.items():
            for job in version['selected_jobs']:
                training = ['runs/train/'+job+'/'+part for part in ['completion.json','logs.jsonl','provenance.json',*(f'validation_epoch_{e:04d}.json' for e in range(1,6))]]
                curves = ['runs/curves/'+job+'/'+part for part in ['curve_receipt.json','epoch_overview.png','epoch_overview.pdf',job+'/steps.csv',job+'/descriptor_components_steps.csv']]
                absent = [n for n in training+curves if n not in members]
                if absent:
                    missing[job] = absent
                problems = []
                for name in training+curves:
                    historical = 'history/'+label+'/'+name.removeprefix('runs/')
                    if aliases.get(historical) != name:
                        problems.append(historical)
                if problems:
                    origin_problems[job] = problems
        completion = read_json(Path(inventory.spec['merged_runs'])/'completion.json')
        if completion.get('analysis_sha256') and completion['analysis_sha256'] != members['runs/final_results/analysis.json']['sha256']:
            raise ValueError('Coordinator and archived analysis SHA-256 differ')
        if completion.get('task_receipt_sha256') and completion['task_receipt_sha256'] != members['runs/task_receipts.json']['sha256']:
            raise ValueError('Coordinator and archived task receipts SHA-256 differ')
        analysis_complete = completion.get('report_complete')
        if not isinstance(analysis_complete, bool):
            # Only older terminal markers lack the scalar result. In production
            # the coordinator's analysis SHA binds this flag to the streamed bytes.
            analysis_complete = read_json(Path(inventory.spec['report'])/'analysis.json').get('complete')
        if analysis_complete and (missing or origin_problems):
            raise ValueError(f'Complete analysis has missing or unverified parent artifacts: {sorted(missing.keys() | origin_problems.keys())}')
        manifest = dict(schema=1, spec_sha256=spec_hash, report_entrypoint='runs/final_results/RESULTS_ZH.html',
                        original_files_preserved=True, includes_model_weights=False, files=members,
                        versions=inventory.versions, identical_parent_files_by_archive_member=aliases,
                        divergent_parent_files=divergent, missing_training_or_curve_artifacts=missing,
                        selected_parent_training_or_curve_problems=origin_problems,
                        terminal_status=completion.get('status'),
                        analysis_complete=analysis_complete)
        payload = json_bytes(manifest)
        bundle.writestr(MANIFEST, payload)
    return manifest, payload


def verify_archive(archive, manifest_bytes=None, verify_members=True):
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive member')
        embedded = bundle.read(MANIFEST)
        if manifest_bytes is not None and manifest_bytes != embedded:
            raise ValueError('Independent download manifest differs from archive')
        manifest = json.loads(embedded)
        files = manifest['files']
        if set(names) != set(files) | {MANIFEST}:
            raise ValueError('Archive and manifest membership differ')
        for name, item in files.items():
            archive_name(name)
            if bundle.getinfo(name).file_size != item['bytes']:
                raise ValueError(f'Member size mismatch: {name}')
            if verify_members:
                with bundle.open(name) as stream:
                    if digest_stream(stream) != item:
                        raise ValueError(f'Member digest mismatch: {name}')
        if any(target not in files for target in manifest['identical_parent_files_by_archive_member'].values()):
            raise ValueError('Historical provenance points to an absent member')
        return manifest, embedded


def receipt_for(archive, manifest, payload):
    return dict(archive=ARCHIVE, **digest_file(archive), archive_members=len(manifest['files'])+1,
                download_manifest=MANIFEST, download_manifest_sha256=hashlib.sha256(payload).hexdigest(),
                spec_sha256=manifest['spec_sha256'], terminal_status=manifest['terminal_status'],
                analysis_complete=manifest['analysis_complete'], includes_model_weights=False)


def verify(output):
    output = Path(output)
    payload = (output/MANIFEST).read_bytes()
    manifest, _ = verify_archive(output/ARCHIVE, payload)
    observed = receipt_for(output/ARCHIVE, manifest, payload)
    if read_json(output/RECEIPT) != observed:
        raise ValueError('Independent package receipt mismatch')
    return observed


def build(spec, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec_hash = hashlib.sha256(json_bytes(spec)).hexdigest()
    if spec.get('schema') != 1:
        raise ValueError('Unsupported bundle spec')
    with output_lock(output):
        archive = output/ARCHIVE
        if (output/RECEIPT).exists():
            receipt = verify(output)
            if receipt['spec_sha256'] != spec_hash:
                raise FileExistsError('Verified package belongs to a different spec')
            return receipt
        if archive.exists():
            # Crash after atomic zip rename, before writing independent sidecars.
            manifest, payload = verify_archive(archive)
        else:
            stage = output/'.portable_bundle.stage'
            stage.mkdir(exist_ok=True)
            temporary = stage/'archive.part'
            inventory = Inventory(spec).collect(output)
            manifest, payload = archive_inventory(temporary, inventory, spec_hash)
            verify_archive(temporary, payload)
            with temporary.open('r+b') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, archive)
            stage.rmdir()
        if manifest['spec_sha256'] != spec_hash:
            raise FileExistsError('Existing archive belongs to a different frozen spec')
        receipt = receipt_for(archive, manifest, payload)
        atomic_bytes(output/MANIFEST, payload)
        atomic_bytes(output/RECEIPT, json_bytes(receipt))
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('make-spec')
    for name in ('merged-runs','original-runs','original-source','original-protocol','recovery-runs','recovery-source','recovery-protocol','write'):
        make.add_argument('--'+name, required=True)
    make.add_argument('--evidence', action='append', default=[], metavar='TAG=PATH')
    make.add_argument('--input', action='append', default=[], metavar='FILE')
    pack = sub.add_parser('build')
    pack.add_argument('--spec', required=True)
    pack.add_argument('--output', required=True)
    check = sub.add_parser('verify')
    check.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.action == 'make-spec':
        spec = make_spec(args)
        print(json.dumps({label:len(v['jobs']) for label,v in spec['versions'].items()}))
    else:
        result = build(read_json(args.spec), args.output) if args.action == 'build' else verify(args.output)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
