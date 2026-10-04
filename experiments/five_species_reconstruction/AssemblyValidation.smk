from pathlib import Path
BASE=Path(workflow.basedir)
PAIRS={'ECOLI_25_vs_100':('ECOLI','00','02'),'PAER_25_vs_100':('PAER','00','02'),'PAER_50_vs_100':('PAER','01','02'),'ABAU_25_vs_100':('ABAU','00','02'),'ABAU_50_vs_100':('ABAU','01','02'),'KPNEU_25_vs_50':('KPNEU','00','01')}
MINIMAP='/tmp/amr-flye-2.9.6/bin/flye-minimap2'
rule all:
 input: expand('{pair}/metrics.json',pair=PAIRS)
rule align:
 input:
  ref=lambda w:str(BASE/'results'/PAIRS[w.pair][0]/PAIRS[w.pair][2]/'assembly.fasta'),
  query=lambda w:str(BASE/'results'/PAIRS[w.pair][0]/PAIRS[w.pair][1]/'assembly.fasta')
 output: '{pair}/alignment.paf'
 log: '{pair}/alignment.log'
 threads: 1
 shell: '{MINIMAP:q} -x asm5 -c --secondary=no -t 1 {input.ref:q} {input.query:q} > {output:q} 2> {log:q}'
rule metrics:
 input:
  ref=lambda w:str(BASE/'results'/PAIRS[w.pair][0]/PAIRS[w.pair][2]/'assembly.fasta'),
  query=lambda w:str(BASE/'results'/PAIRS[w.pair][0]/PAIRS[w.pair][1]/'assembly.fasta'),
  paf='{pair}/alignment.paf'
 output: '{pair}/metrics.json'
 params: script=str(BASE/'summarize.py')
 shell: 'python3 {params.script:q} {input.ref:q} {input.query:q} {input.paf:q} > {output:q}'
