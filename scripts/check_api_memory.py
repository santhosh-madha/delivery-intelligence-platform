"""Test the Docker API with 512 MiB and no swap against a disposable database."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
EVENT = dict(platform_order_time=1666407600, order_push_time=1666407660,
             grab_time=1666407720, poi_id=1, da_id=0, courier_id=2, is_prebook=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'artifacts/models/87a38ec8f34d4ed0a47437a3b1e120ee')
    args = parser.parse_args()
    bundle = args.model_dir.resolve()
    if not (bundle/'model.cbm').is_file():
        raise FileNotFoundError(bundle/'model.cbm')
    env = dict(POSTGRES_DB='memory_test', POSTGRES_USER='memory_test',
               POSTGRES_PASSWORD='disposable_test_only')
    config = {'services': {
        'db': {'image': 'postgres:18', 'environment': env,
               'tmpfs': ['/var/lib/postgresql'],
               'healthcheck': {'test': ['CMD-SHELL', 'pg_isready -U memory_test -d memory_test'],
                               'interval': '2s', 'timeout': '3s', 'retries': 30}},
        'api': {'build': {'context': str(ROOT)}, 'mem_limit': '512m',
                'memswap_limit': '512m', 'cpus': 1,
                'environment': {**env, 'POSTGRES_HOST': 'db', 'POSTGRES_PORT': '5432', 'ETA_MODEL_DIR': '/app/model'},
                'ports': ['127.0.0.1:18080:8000'],
                'volumes': [{'type':'bind', 'source':str(bundle), 'target':'/app/model', 'read_only':True,
                             'bind': {'create_host_path':False}}],
                'depends_on': {'db': {'condition':'service_healthy'}},
                'healthcheck': {'test':['CMD','python','-c',
                    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).close()"],
                    'interval':'3s','timeout':'6s','retries':30,'start_period':'15s'}},
    }}
    with tempfile.TemporaryDirectory(prefix='eta-memory-') as temp:
        path = Path(temp)/'compose.json'
        path.write_text(json.dumps(config))
        compose = ['docker','compose','-p','delivery-eta-memory-check','-f',str(path)]
        def command(*parts, capture=False, timeout=180):
            return subprocess.run(compose+list(parts), check=True, text=True,
                                  capture_output=capture, timeout=timeout)
        def memory():
            result = command('exec','-T','api','sh','-c',
                'cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory.events', capture=True)
            lines = result.stdout.splitlines()
            return {'current_bytes':int(lines[0]), 'peak_bytes':int(lines[1]),
                    'events':{k:int(v) for k,v in (line.split() for line in lines[2:])}}
        def request(_):
            start=perf_counter()
            req=urllib.request.Request('http://127.0.0.1:18080/predict',
                data=json.dumps(EVENT).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=60) as response:
                body=json.load(response)
                assert response.status==200
                assert abs(body['predicted_duration_min']-32.2649262408378)<1e-8
            return perf_counter()-start
        report = {'memory_limit_bytes':512*1024*1024,'swap_allowed_bytes':0,
                  'cpu_limit':1,'model_version':bundle.name,
                  'limitations':'Local Docker architecture, one CPU, synthetic requests. Memory smoke test, not full Render CPU/network/cold-start equivalence or sustained-load certification.'}
        try:
            command('build','api',timeout=600)
            command('up','-d','--wait','db')
            command('run','--rm','--no-deps','api','alembic','upgrade','head')
            start=perf_counter()
            command('up','-d','--wait','api')
            report['startup_until_healthy_seconds']=perf_counter()-start
            report['startup_memory']=memory()
            cid=command('ps','-q','api',capture=True).stdout.strip()
            inspection=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
            assert inspection['HostConfig']['Memory']==512*1024*1024
            assert inspection['HostConfig']['MemorySwap']==512*1024*1024
            report['architecture']=subprocess.check_output(['docker','info','--format','{{.Architecture}}'],text=True).strip()
            times=[request(i) for i in range(50)]
            report['sequential']={'successful_requests':50,'mean_seconds':sum(times)/len(times)}
            with ThreadPoolExecutor(max_workers=4) as pool:
                times=list(pool.map(request,range(40)))
            report['concurrent']={'workers':4,'successful_requests':40,'mean_seconds':sum(times)/len(times)}
            report['final_memory']=memory()
            inspection=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
            report['oom_killed']=inspection['State']['OOMKilled']
            assert not report['oom_killed']
            assert report['final_memory']['events'].get('oom',0)==0
            assert report['final_memory']['events'].get('oom_kill',0)==0
            rows=command('exec','-T','db','psql','-U','memory_test','-d','memory_test','-tAc',
                         'SELECT count(*) FROM predictions;',capture=True).stdout.strip()
            assert int(rows)==90
            report['persisted_demo_predictions']=int(rows)
            report['status']='passed'
        finally:
            command('down','--volumes',timeout=60)
        out=ROOT/'artifacts/memory_checks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out.mkdir(parents=True,exist_ok=False)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'output':str(out),**report},indent=2))


if __name__=='__main__':
    main()
