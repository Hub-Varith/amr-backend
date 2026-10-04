from pathlib import Path
BASE=Path(workflow.basedir)
REF=str(BASE.parent/'ecoli_reconstruction/reference.fasta')
PYTHON='/Library/Frameworks/Python.framework/Versions/3.14/bin/python3'
MINIMAP='/tmp/amr-flye-2.9.6/bin/flye-minimap2'
rule all:
    input: expand('{level}/metrics.json',level=['00','02'])
rule align:
    input: ref=REF,asm=lambda w:str(BASE/f'results/ECOLI/{w.level}/assembly.fasta')
    output: '{level}/alignment.paf'
    log: '{level}/alignment.log'
    threads: 1
    shell: '{MINIMAP:q} -x asm5 -c --secondary=no -t 1 {input.ref:q} {input.asm:q} > {output:q} 2> {log:q}'
rule metrics:
    input: ref=REF,asm=lambda w:str(BASE/f'results/ECOLI/{w.level}/assembly.fasta'),paf='{level}/alignment.paf'
    output: '{level}/metrics.json'
    params: script=str(BASE/'summarize.py')
    shell: '{PYTHON:q} {params.script:q} {input.ref:q} {input.asm:q} {input.paf:q} > {output:q}'
