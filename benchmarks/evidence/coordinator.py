"""Resume the 157 frozen evaluations, using the exact engine that trained each model.

Linux deployment. `preflight` uses CPU only and permits an unfinished Q80 queue.
`run` waits for both source queues, verifies every training and checkpoint, runs
the frozen commands, then produces the report and verified portable delivery.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


EXPERIMENTS = Path(os.environ.get('WIRESEG_EXPERIMENTS', 'experiments')).resolve()
VERSIONS = Path(os.environ.get('WIRESEG_VERSIONS', 'cable_seg/versions'))
R2 = 'pilot_evidence_20261031_r2'
RECOVERY = 'pilot_evidence_20261102_q80fix'
KINDS = {'synthetic':84, 'robustness':30, 'benchmark':13, 'metrics':12, 'external':18}
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write('\n')
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else block_hash(stream)


def block_hash(stream):
    result=hashlib.sha256()
    for data in iter(lambda:stream.read(8*1024**2),b''): result.update(data)
    return result.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def same_or_write(path,obj):
    if path.exists():
        if read(path)!=obj: raise ValueError(f'Persisted plan/input differs: {path}')
    else: write(path,obj)


def require(condition,message):
    if not condition: raise ValueError(message)


def under(path, parent):
    relative=Path(path).relative_to(parent)
    require(relative.parts and '..' not in relative.parts,'Unsafe relative path: '+str(path))
    return relative


def relocate(protocol, old_root, new_root):
    """Only rewrite additional evaluation checkpoint/output/report paths."""
    result=copy.deepcopy(protocol)
    for item in result['additional_evaluations']:
        require(item['module'] in ('external_data','scalability'),'Unexpected additional evaluator')
        args=item['arguments']; expected_ckpt=Path(old_root)/'train'/item['job']/(item['tag']+'.pt')
        require(Path(args[args.index('--checkpoint')+1])==expected_ckpt,'Unexpected frozen checkpoint path')
        old_dest=Path(args[args.index('--output')+1]); report=Path(item['report'])
        require(report.parent==old_dest and under(old_dest,old_root).parts[0] in ('external','real'),'Unexpected frozen evaluation destination')
        for flag in ('--checkpoint','--output'):
            index=args.index(flag)+1; args[index]=str(Path(new_root)/under(args[index],old_root))
        item['report']=str(Path(new_root)/under(report,old_root))
    return result


def commands(protocol,expected,root):
    jobs={j['id']:j for j in protocol['jobs']}
    additional={item['id']:item for item in protocol['additional_evaluations']}
    result=[]
    for item in expected:
        job=jobs[item['job']]; tag=item.get('tag') or 'best'; dest=Path(item['report']).parent
        checkpoint=Path(root)/'train'/job['id']/(tag+'.pt')
        base=['--checkpoint',str(checkpoint),'--manifest',protocol['test_manifest'],'--data-root',protocol['test_data_root']]
        if item['kind']=='synthetic': module,args='evaluate',base+['--split','test','--output',str(dest),'--device','cuda','--allow-condition','--identity-diagnostics']
        elif item['kind']=='robustness': module,args='robustness',base+['--output',str(dest),'--device','cuda','--count','300']
        elif item['kind']=='benchmark': module,args='benchmark',base+['--output',str(dest)]
        else:
            source=additional[item['id']]; module,args=source['module'],source['arguments']
        result.append(dict(**item,module=module,arguments=args,engine=RECOVERY if job['stage']=='native80' else R2))
    require(Counter(t['kind'] for t in result)==KINDS and len({t['id'] for t in result})==157,'Expected 84/30/13/12/18 registered evaluations')
    require(len({t['report'] for t in result})==157,'Overlapping evaluation destinations')
    return result


def link_files(source,destination):
    """Real directories with individual file links so recursive packagers see every curve/log."""
    require(source.is_dir(),f'Missing input directory: {source}')
    for path in source.rglob('*'):
        if not path.is_file(): continue
        target=destination/path.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink(): require(target.resolve()==path.resolve(),f'Wrong evidence link: {target}')
        else:
            require(not target.exists(),f'Refusing to replace existing file: {target}')
            target.symlink_to(path.resolve())


def inventory(folder):
    return {path.relative_to(folder).as_posix():digest(path) for path in sorted(folder.rglob('*')) if path.is_file()}


def set_engine(source):
    source=Path(source).resolve();sys.path.insert(0,str(source))
    import sam_path_v1
    require(Path(sam_path_v1.__file__).resolve().parent==source/'sam_path_v1','Wrong Python engine imported')


def engine_env(source):
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['MPLBACKEND']='Agg'
    env.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    env['PYTHONPATH']=str(source)+os.pathsep+env.get('PYTHONPATH','')
    return env


def cpu_gate(spec):
    """Called in a fresh CPU-only process for each frozen source tree."""
    set_engine(spec['source'])
    from sam_path_v1.checkpoint import read_checkpoint, verify_engine_source
    from sam_path_v1.train import TrainConfig
    from sam_path_v1.evidence_run import verify_log_membership
    from sam_path_v1.full_run import verify_training
    from sam_path_v1.evidence_report import audit_training
    from dataclasses import asdict
    audits=[]
    for job in spec['jobs']:
        root=Path(spec['runs']); folder=root/'train'/job['id']
        config=asdict(TrainConfig(**job['config']))
        require(read(job['config_path'])==job['config'] and read(folder/'config.json')==config,'Training config changed: '+job['id'])
        if spec['full']:
            verify_training(folder,config,{'train':2000,'validation':300})
            rows=read(config['manifest'])['records']
            verify_log_membership(folder,config,{r['id'] for r in rows if r['split']=='train'},
                                  {r['id'] for r in rows if r['split']=='validation'})
            audit=audit_training(job,root)
            require(audit['complete'],f'Training/curve audit failed: {job["id"]} {audit["problems"]}')
        else: audit={'job':job['id']}
        checks={}
        for tag in ('best','last'):
            path=folder/(tag+'.pt'); payload=read_checkpoint(path)
            verify_engine_source(payload); p=payload['progress']
            require(payload['model_config']==config['model'] and payload['train_config']==config and
                    1<=p['epoch']<=5 and p['successful_updates']==2000*p['epoch'] and p['amp_skips']==0
                    and (tag!='last' or p['epoch']==5),'Checkpoint/progress mismatch: '+str(path))
            checks[tag]=dict(sha256=digest(path),updates=p['successful_updates'],epoch=p['epoch'],engine_source_sha256=payload['provenance']['engine_source_sha256'])
            del payload
        audits.append(dict(**audit,checkpoints=checks))
    write(spec['receipt'],audits)


def execute_child(spec):
    # This wrapper outlives a disconnected supervisor and records the evaluator's
    # real exit status. Its inherited flock also excludes duplicate GPU workers.
    inherited=tuple(spec.get('lock_fds',[]))
    with Path(spec['log']).open('xb') as stream:
        code=subprocess.run([sys.executable,'-B','-u','-m','sam_path_v1.'+spec['module'],*spec['arguments']],
            cwd=spec['source'],env=engine_env(spec['source']),stdout=stream,stderr=subprocess.STDOUT,
            pass_fds=inherited).returncode
    write(spec['exit_receipt'],dict(code=code,signature=spec['signature']))


def file_check(paths):
    for filename,expected in paths.items():
        require(Path(filename).is_file() and digest(filename)==expected,'Frozen input changed: '+filename)


def validate_data(protocol):
    """Validate all intended RGB/annotation filenames, without loading scores."""
    from sam_path_v1.data import resolved_record_path
    def members(path,root,expected):
        records=[r for r in read(path)['records'] if r['split']=='test']
        require(len(records)==expected==len({r['id'] for r in records}),'Data count/ID mismatch: '+str(path))
        for row in records:
            for key in ('image','masks'):
                f=resolved_record_path(root,row[key]);require(f.is_file(),'Missing data file: '+str(f))
    members(protocol['test_manifest'],protocol['test_data_root'],1800)
    audited=set()
    for task in protocol['additional_evaluations']:
        args=task['arguments']
        if task['module']=='scalability':
            path=args[args.index('--manifest')+1];root=args[args.index('--data-root')+1]
            if path not in audited: members(path,root,task['expected_images']);audited.add(path)
        else:
            root=Path(args[args.index('--audit-root')+1]); path=root/'audit.json'
            if path in audited:continue
            audited.add(path); records=read(path)['records']
            require(len(records)==110==len({r['id'] for r in records}),'HANDLOOM manifest count/IDs')
            for row in records:
                require(row.get('status')=='rgb_and_numeric_trace_verified','HANDLOOM audit failed')
                for key in ('rgb_path','trace_path'):
                    require((root/row[key]).is_file(),'Missing HANDLOOM payload: '+str(row[key]))


class Coordinator:
    def __init__(self,args):
        self.root=args.output.resolve();self.args=args; self.lock_fd=None;self.gpu_lock_fd=None
        self.parent=args.parent_runs.resolve();self.recovery=args.recovery_runs.resolve()
        require(not any(self.root==p or p in self.root.parents or self.root in p.parents for p in (self.parent,self.recovery)), 'Output must be isolated from both original queues')
        self.sources={R2:args.r2_source.resolve(),RECOVERY:args.recovery_source.resolve()}
        set_engine(self.sources[R2]);from sam_path_v1.evidence_run import EvidenceQueue
        self.original=read(self.parent/'protocol.json'); self.protocol=relocate(self.original,self.parent,self.root)
        self.queue=EvidenceQueue(self.protocol,self.root);self.tasks=commands(self.protocol,self.queue.expected_evaluations,self.root)
        self.jobs={j['id']:j for j in self.protocol['jobs']}
        require(len(self.jobs)==42 and sum(j['stage']=='native80' for j in self.jobs.values())==3,'Invalid frozen training matrix')
        self.test_ids={r['id'] for r in read(self.protocol['test_manifest'])['records'] if r['split']=='test'}

    def status(self,phase,**extra):
        write(self.root/'status.json',dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**extra))

    def preflight(self,full=False):
        root=self.root;root.mkdir(parents=True,exist_ok=True)
        original_done=read(self.parent/'completion.json'); completed={x['id'] for x in original_done['training']}
        require(original_done['status']=='failed' and completed=={j['id'] for j in self.jobs.values() if j['stage']!='native80'},'Expected failed r2 with 39 completed jobs')
        fix=read(self.recovery/'protocol.json')
        require({j['id'] for j in fix['jobs']}==set(self.jobs)-completed and all(j['config']==self.jobs[j['id']]['config'] for j in fix['jobs']),'Q80 protocol lineage mismatch')
        self.fix_jobs={j['id']:j for j in fix['jobs']}
        inputs={}
        for version,folder in ((R2,self.parent),(RECOVERY,self.recovery)):
            freeze=folder/'freeze.json'; recorded=read(freeze)['frozen_files']
            source=self.sources[version]/'sam_path_v1'
            require(str((source/'evaluate.py').resolve()) in recorded,'Evaluator not included in frozen source receipt: '+str(source))
            for path,value in recorded.items():
                require(path not in inputs or inputs[path]==value,'Conflicting source/input hash: '+path)
                inputs[path]=value
        file_check(inputs)
        self.engine_files={version:{filename:value for filename,value in inputs.items()
            if Path(filename).is_relative_to(source/'sam_path_v1')} for version,source in self.sources.items()}
        require(digest(self.protocol['test_manifest'])==self.protocol['test_manifest_sha256'],'Fixed test manifest hash mismatch')
        validate_data(self.protocol)
        for p in self.protocol['frozen_inputs']: require(Path(p).is_file(),'Missing frozen input: '+p)
        old=read(self.parent/'expected_evaluations.json')
        normalized=[{**row,'report':str(root/under(row['report'],self.parent))} for row in old]
        require(normalized==self.queue.expected_evaluations,'Relocated expected tasks diverge from frozen 157-task roster')
        same_or_write(root/'protocol.json',self.protocol)
        same_or_write(root/'expected_evaluations.json',self.queue.expected_evaluations)
        same_or_write(root/'task_plan.json',self.tasks)
        done_path=self.recovery/'completion.json'; fix_done=read(done_path) if done_path.is_file() else None
        if fix_done:require(fix_done.get('status')=='completed','Q80 recovery has failed; inspect original completion')
        if full:
            for helper in ('synthesis.py','portable_bundle.py','bounded_report.py'):
                require((HERE/helper).is_file(),'Missing final report/package helper: '+helper)
            require(fix_done and {x['id'] for x in fix_done['audits']}==set(self.fix_jobs) and
                fix_done['parent_39_unchanged'] and fix_done['exhaustive_mosaic_preflight_passed'],'Q80 completion/recipe audit missing')
            for proof in fix_done['audits']:
                require(proof['successful_updates']==10000 and proof['validation_epochs']==5 and proof['recipes']['checked']==10000 and
                        proof['recipes']['exact_preflight_correspondence'],'Incomplete Q80 recipe validation')
                for tag in ('best','last'):
                    require(digest(self.recovery/'train'/proof['id']/(tag+'.pt'))==proof[tag+'_sha256'],'Q80 checkpoint changed')
            file_check(read(self.recovery/'parent_preservation.json')['sha256'])
            file_check(read(self.recovery/'freeze_after_data_admission.json')['frozen_files'])
            bundle=read(self.recovery/'package_receipt.json');require(digest(bundle['archive'])==bundle['sha256'],'Q80 training archive hash mismatch')
        audits=[]
        for version,jobs,folder in ((R2,[self.jobs[x] for x in sorted(completed)],self.parent),
                                    (RECOVERY,list(self.fix_jobs.values()),self.recovery)):
            available=[j for j in jobs if (folder/'curves'/j['id']/'curve_receipt.json').is_file()]
            require(not full or len(available)==len(jobs),'Missing completed training/curves')
            spec=dict(source=str(self.sources[version]),runs=str(folder),jobs=available,full=full,
                      receipt=str(root/'checks'/('checkpoints_'+version+'.json')))
            path=root/'checks'/('gate_'+version+'.json');write(path,spec)
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_gate',str(path)],check=True,
                env={**engine_env(self.sources[version]),'CUDA_VISIBLE_DEVICES':''})
            audits+=read(spec['receipt'])
            if full:
                for job in jobs:
                    for name in ('train','curves'):link_files(folder/name/job['id'],root/name/job['id'])
        self.checkpoints={a['job']:a['checkpoints'] for a in audits}
        verified={'full_gate':full,'checkpoints':len(audits)*2,'training_jobs':len(audits),'frozen_files':len(inputs),
                  'registered_tasks':dict(Counter(x['kind'] for x in self.tasks)), 'parent_completion_sha256':digest(self.parent/'completion.json'),
                  'recovery_completion_sha256':digest(done_path) if fix_done else None}
        write(root/'checks'/('training_gate.json' if full else 'preflight.json'),verified)
        return verified

    def signature(self,task):
        tag=task.get('tag') or 'best';checkpoint=self.checkpoints[task['job']][tag]
        return fingerprint(dict(task=task,checkpoint=checkpoint,protocol_sha256=digest(self.root/'protocol.json')))

    def verify_task(self,task,report):
        from sam_path_v1.evidence_run import verify_test_report,exact_members
        from sam_path_v1.benchmark import benchmark_diagnostic_problems
        tag=task.get('tag') or 'best'; expected=self.checkpoints[task['job']][tag]['sha256']
        kind,args=task['kind'],task['arguments']
        if kind=='synthetic':return verify_test_report(report,self.test_ids,self.protocol['test_manifest_sha256'],expected,self.jobs[task['job']]['config'],tag)
        require(report.get('checkpoint_sha256')==expected,'Evaluated checkpoint SHA mismatch: '+task['id'])
        if kind=='robustness':
            require(report.get('manifest_sha256')==self.protocol['test_manifest_sha256'],'Occlusion manifest mismatch')
            return True # EvidenceQueue.record_task verifies five frozen severity rosters.
        if kind=='benchmark':
            require(report.get('manifest_sha256')==self.protocol['test_manifest_sha256'],'Benchmark manifest mismatch')
            return not benchmark_diagnostic_problems(report,self.jobs[task['job']]['variant'])
        if kind=='external':records=read(Path(args[args.index('--audit-root')+1])/'audit.json')['records']
        else:
            manifest=Path(args[args.index('--manifest')+1]);require(report.get('manifest_sha256')==digest(manifest),'Native manifest mismatch')
            records=[r for r in read(manifest)['records'] if r['split']=='test']
        exact_members(report['per_image'],{r['id'] for r in records},task['id'])
        return True

    def run_task(self,task,allow_execute=True):
        ident=task['id']; sig=self.signature(task); destination=Path(task['report']).parent
        history=self.root/'attempts'/ident; history.mkdir(parents=True,exist_ok=True)
        earlier=sorted(p for p in history.iterdir() if p.is_dir() and p.name.isdigit())
        previous=earlier[-1] if earlier else None
        # Recover completed exit receipts, including a child whose launcher died
        # before registration. Never overwrite an evaluator's output directory.
        if previous and (previous/'exit.json').is_file():
            code=read(previous/'exit.json'); require(code['signature']==sig,'Changed inputs on resume: '+ident)
            if code['code'] in (0,2):
                prior=None
                if (previous/'receipt.json').is_file():
                    prior=read(previous/'receipt.json'); require(prior['signature']==sig and inventory(destination)==prior['files'],'Prior artifacts changed: '+ident)
                if not self.args.retry_failed or prior and prior['complete']:
                    return self.register(task,previous,code['code'],sig)
                if prior is None and code['code']==0:
                    try:
                        recovered=self.register(task,previous,code['code'],sig)
                        if recovered['complete']:return recovered
                        self.queue.task_receipts.remove(recovered); self.queue.write_task_receipts()
                    except (OSError,ValueError,KeyError,TypeError) as exc:
                        write(previous/'invalid_output.json',dict(error=str(exc),signature=sig))
        require(allow_execute,f'Finalize requires a registered evaluation with exit 0/2: {ident}')
        file_check(self.engine_files[task['engine']])
        if destination.exists():
            require(previous is not None,'Unowned evaluation destination exists: '+str(destination))
            require(not destination.is_symlink() and destination.is_dir(),'Refusing to move non-owned destination')
            require(destination.resolve().is_relative_to(self.root),'Output escapes isolated root')
            archive=previous/'preserved_output';require(not archive.exists(),'Attempt archive collision');destination.rename(archive)
        attempt=history/f'{len(earlier)+1:04d}';attempt.mkdir()
        inherited=tuple(fd for fd in (self.lock_fd,self.gpu_lock_fd) if fd is not None)
        spec=dict(signature=sig,log=str(attempt/'stdout.log'),exit_receipt=str(attempt/'exit.json'),lock_fds=inherited,
                  source=str(self.sources[task['engine']]),module=task['module'],arguments=task['arguments'])
        write(attempt/'invocation.json',spec);self.status('evaluating',task=ident,attempt=attempt.name,registered=len(self.queue.task_receipts))
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_child',str(attempt/'invocation.json')],
            check=True,pass_fds=inherited,env=engine_env(self.sources[task['engine']]))
        code=read(attempt/'exit.json')['code']
        file_check(self.engine_files[task['engine']])
        require(code in (0,2),f'{ident} exited {code}, retained {attempt}/stdout.log')
        return self.register(task,attempt,code,sig)

    def register(self,task,attempt,code,sig):
        path=Path(task['report']);override=self.verify_task(task,read(path)) if path.is_file() else False
        row=self.queue.record_task(task['id'],code,override)
        if not (attempt/'receipt.json').exists():
            write(attempt/'receipt.json',dict(**row,signature=sig,files=inventory(path.parent)))
        return row

    def finalize(self):
        receipts=self.queue.write_task_receipts();require(not receipts['missing_expected'],'Missing fixed evaluations')
        out=self.root/'final_results'; protocol=self.root/'protocol.json'
        self.status('reporting',registered=len(receipts['tasks']))
        # A fresh report directory is required by the frozen renderer. A partial
        # or explicitly superseded report remains available for inspection.
        state=fingerprint([(r['id'],r['sha256'],r['complete']) for r in receipts['tasks']])
        render_key=out/'coordinator_render.json'
        if not (render_key.is_file() and read(render_key).get('state')==state and (out/'analysis.json').is_file()):
            if out.exists():
                archive=self.root/'prior_reports'/uuid.uuid4().hex;archive.parent.mkdir(exist_ok=True);out.rename(archive)
            base_args=['--protocol',str(protocol),'--runs',str(self.root),'--output',str(out)]
            bounded=HERE/'bounded_report.py'
            if bounded.is_file():argv=[sys.executable,'-B',str(bounded),*base_args,'--engine',str(self.sources[R2])]
            else:argv=[sys.executable,'-B','-m','sam_path_v1.evidence_report',*base_args]
            subprocess.run(argv,cwd=self.sources[R2],env=engine_env(self.sources[R2]),check=True)
            write(render_key,{'state':state})
        subprocess.run([sys.executable,'-B',str(HERE/'synthesis.py'),'--output',str(out)],check=True)
        analysis=read(out/'analysis.json');complete=receipts['complete'] and analysis.get('complete') is True
        terminal=dict(status='completed' if complete else 'completed_with_evaluation_failures',
            successful_training=analysis.get('successful_training'),training_gate=read(self.root/'checks'/'training_gate.json'),
            registered=len(receipts['tasks']),failures=[r['id'] for r in receipts['tasks'] if not r['complete']],
            report_complete=analysis.get('complete'),task_receipt_sha256=digest(self.root/'task_receipts.json'),
            analysis_sha256=digest(out/'analysis.json'),origin='39 verified r2 trainings and 3 verified Q80 recovery trainings')
        pending={**terminal,'status':'evaluations_finished_pending_package','evaluation_status':terminal['status'],
                 'delivery_state':'awaiting_verified_archive','scope':'Evaluation-stage snapshot in the archive; external delivery receipt records verification.'}
        write(self.root/'evaluation_completion.json',pending)
        write(self.root/'completion.json',pending)
        bundle=HERE/'portable_bundle.py';spec=self.root/'bundle_spec.json'
        make=[sys.executable,'-B',str(bundle),'make-spec','--merged-runs',str(self.root),
              '--original-runs',str(self.parent),'--original-source',str(self.sources[R2]/'sam_path_v1'),
              '--original-protocol',str(self.parent/'protocol.json'),'--recovery-runs',str(self.recovery),
              '--recovery-source',str(self.sources[RECOVERY]/'sam_path_v1'),'--recovery-protocol',str(self.recovery/'protocol.json'),
              '--evidence','failed_q80_parent='+str(self.parent/'completion.json'),
              '--evidence','original_failed_admission='+str(self.parent.parent/'runs'),'--write',str(spec)]
        for name in ('coordinator.py','synthesis.py','bounded_report.py','portable_bundle.py'):
            make+=['--evidence','eval_'+Path(name).stem+'='+str(HERE/name)]
        for name in ('prepare_server.py','queue_stdout.log','source_transport_receipt.json','parent_source_sha256.json','deployment_receipt.json'):
            p=self.recovery.parent/name
            if p.is_file():make+=['--evidence','q80_'+p.stem+'='+str(p)]
        for path in self.protocol.get('frozen_inputs',[]):make+=['--input',path]
        subprocess.run(make,check=True)
        subprocess.run([sys.executable,'-B',str(bundle),'build','--spec',str(spec),'--output',str(out)],check=True)
        subprocess.run([sys.executable,'-B',str(bundle),'verify','--output',str(out)],check=True)
        verified={**terminal,'delivery_state':'archive_verified','package_receipt_sha256':digest(out/'package_receipt.json'),
                  'archive_sha256':read(out/'package_receipt.json')['sha256']}
        write(self.root/'delivery_completion.json',verified)
        write(self.root/'completion.json',verified)
        self.status(terminal['status'],package_verified=True);return complete

    def run(self,phase):
        if phase=='preflight':return self.preflight()
        if phase=='run':
            missing_launcher=0
            while not (self.recovery/'package_receipt.json').is_file():
                terminal=self.recovery/'completion.json'
                if terminal.is_file() and read(terminal).get('status')=='failed':raise RuntimeError('Q80 recovery failed')
                status=self.recovery/'status.json'; pid=read(status).get('launcher_pid') if status.is_file() else None
                proc=Path('/proc')/str(pid)/'cmdline'
                if Path('/proc/self').exists():
                    try: alive=pid and b'q80_recovery' in proc.read_bytes()
                    except FileNotFoundError: alive=False
                    missing_launcher=0 if alive else missing_launcher+1
                    if missing_launcher>=2 and not (self.recovery/'package_receipt.json').is_file():
                        raise RuntimeError(f'Q80 supervisor PID {pid} is absent without its required package receipt')
                self.status('waiting_for_q80_training_package');time.sleep(self.args.poll_seconds)
            # The recovery can still hold the project lock when its receipt first
            # becomes visible. Serialize with every other project GPU supervisor.
            import fcntl
            self.status('waiting_for_project_gpu_lock')
            with self.args.gpu_lock.open('a+') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX);self.gpu_lock_fd=lock.fileno()
                try:return self.evaluate(phase)
                finally:self.gpu_lock_fd=None
        return self.evaluate(phase)

    def evaluate(self,phase):
        self.preflight(full=True)
        for task in self.tasks:
            if phase=='finalize':
                history=self.root/'attempts'/task['id']
                require(history.is_dir() and any(history.glob('*/exit.json')),'Finalize needs each evaluator exit receipt')
            self.run_task(task,allow_execute=phase=='run')
        return self.finalize()


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('_gate','_child'):
        (cpu_gate if argv[0]=='_gate' else execute_child)(read(argv[1]));return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','run','finalize'))
    parser.add_argument('--output',type=Path,default=EXPERIMENTS/'pilot_evidence_eval_20261102/runs')
    parser.add_argument('--parent-runs',type=Path,default=EXPERIMENTS/'pilot_evidence_20261031/runs_r2')
    parser.add_argument('--recovery-runs',type=Path,default=EXPERIMENTS/RECOVERY/'runs')
    parser.add_argument('--r2-source',type=Path,default=VERSIONS/R2)
    parser.add_argument('--recovery-source',type=Path,default=VERSIONS/RECOVERY)
    parser.add_argument('--gpu-lock',type=Path,default=EXPERIMENTS/'.sam_path_v1.pilot.lock')
    parser.add_argument('--poll-seconds',type=float,default=45)
    parser.add_argument('--retry-failed',action='store_true',help='Explicitly repeat previously scored incomplete tasks; old attempt remains archived')
    args=parser.parse_args(argv);require(args.poll_seconds>0,'Polling interval must be positive')
    coordinator=Coordinator(args);coordinator.root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (coordinator.root/'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);coordinator.lock_fd=lock.fileno()
        try:
            result=coordinator.run(args.phase)
            print(json.dumps(result,ensure_ascii=False),flush=True);return 2 if result is False else 0
        except Exception as exc:
            coordinator.status('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':raise SystemExit(main())
