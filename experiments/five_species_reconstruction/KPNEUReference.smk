from pathlib import Path
BASE=Path(workflow.basedir)
rule all:
 input: expand('{level}/metrics.json',level=['00','01'])
rule align:
 input:
  ref=str(BASE/'assembly_validation/kpneu_reference/reference.fasta'),
  asm=lambda w:str(BASE/'results/KPNEU'/w.level/'assembly.fasta')
 output: '{level}/alignment.paf'
 log: '{level}/alignment.log'
 threads: 1
 shell: '/tmp/amr-flye-2.9.6/bin/flye-minimap2 -x asm5 -c --secondary=no -t 1 {input.ref:q} {input.asm:q} > {output:q} 2> {log:q}'
rule metrics:
 input:
  ref=str(BASE/'assembly_validation/kpneu_reference/reference.fasta'),
  asm=lambda w:str(BASE/'results/KPNEU'/w.level/'assembly.fasta'),
  paf='{level}/alignment.paf'
 output: '{level}/metrics.json'
 params: script=str(BASE/'summarize.py')
 shell: 'python3 {params.script:q} {input.ref:q} {input.asm:q} {input.paf:q} > {output:q}'
