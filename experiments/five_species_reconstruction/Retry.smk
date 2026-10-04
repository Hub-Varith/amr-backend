from pathlib import Path
BASE=Path(workflow.basedir)
rule all:
 input: 'stats.json'
rule assemble:
 input: config['reads']
 output: 'assembly.fasta'
 threads: 1
 log: 'attempt.log'
 params: out=config['out'],size=config['size'],coverage=config['coverage'],guard=str(BASE/'guard.py'),python=config['python'],flye=config['flye']
 shell: '{params.python:q} {params.guard:q} --status guard_status.json --seconds 2700 --max-rss-gb 6 -- {params.python:q} {params.flye:q} --nano-raw {input:q} --genome-size {params.size} --asm-coverage {params.coverage} --threads 1 --out-dir {params.out:q} > {log:q} 2>&1'
rule stats:
 input: 'assembly.fasta'
 output: 'stats.json'
 params: script=str(BASE/'stats.py'),python=config['python']
 shell: '{params.python:q} {params.script:q} {input:q} {output:q}'
