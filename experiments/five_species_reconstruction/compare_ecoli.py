"""Summarize completed side-validation alignments; count differences, not clinical errors."""
import json,re
from pathlib import Path
root=Path(__file__).resolve().parent/'side_validation'
rows=[]
for level,fraction in [('00',25),('02',100)]:
 m=json.loads((root/level/'metrics.json').read_text())
 counts=dict(substitutions=0,inserted_bases=0,deleted_bases=0,alignment_columns=0,matching_bases=0)
 for line in (root/level/'alignment.paf').read_text().splitlines():
  fields=line.split('\t')
  if 'tp:A:P' not in fields[12:]:continue
  tags={t.split(':',2)[0]:t.split(':',2)[2] for t in fields[12:]}
  ops=re.findall(r'(\d+)([MIDNSHP=X])',tags['cg'])
  insertions=sum(int(n) for n,op in ops if op=='I')
  deletions=sum(int(n) for n,op in ops if op=='D')
  substitutions=int(tags['NM'])-insertions-deletions
  assert substitutions>=0
  counts['substitutions']+=substitutions
  counts['inserted_bases']+=insertions;counts['deleted_bases']+=deletions
  counts['alignment_columns']+=int(fields[10]);counts['matching_bases']+=int(fields[9])
 assert counts['alignment_columns']-counts['matching_bases']==sum(counts[k] for k in ['substitutions','inserted_bases','deleted_bases'])
 rows.append(dict(read_percent=fraction,metrics=m,alignment_difference_counts=counts))
(root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# E. coli reference comparison','','Completed alongside the main batch, using one alignment thread. No training or MIC prediction.','','| Reads used | Contigs | Reference alignment breadth | Aligned identity | Substitution differences | Inserted bases | Deleted bases |','|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
 m=row['metrics'];c=row['alignment_difference_counts']
 lines.append(f"| {row['read_percent']}% | {m['contigs']} | {m['reference_covered_percent']:.5f}% | {m['aligned_identity_percent']:.5f}% | {c['substitutions']:,} | {c['inserted_bases']:,} | {c['deleted_bases']:,} |")
lines+=['','Reference alignment breadth counts alignment spans, including internal gaps; it is NOT the percentage of reference bases correctly reconstructed. Difference counts sum primary alignment records and are not deduplicated per genomic position. They are relative to the reference, not independently confirmed sequencing errors. Structural rearrangements and AMR features have not been evaluated.','','Reference: manufacturer ZymoBIOMICS v3 E. coli chromosome and plasmid. This is not verified byte-identical to the historical reference for the ONT tutorial reads. Reference differences may contribute to these counts. See ../ecoli_reconstruction/README.md for provenance.','','Interpretation: using all reads improves aligned identity over 25% in this run, but both retain substantial differences. Neither result establishes suitability for MIC prediction. The current baseline uses a 30x initial-assembly cap and two threads; earlier pilot numbers used different settings and must not be pooled.','','Reproduce from the experiment directory:','```sh','/tmp/amr-benchmark-env/bin/snakemake --snakefile SideValidation.smk --directory side_validation --cores 1','python3 compare_ecoli.py','```','','These are research inputs for future predictions of in-vitro susceptibility, not prescribing advice.','']
(root/'REPORT.md').write_text('\n'.join(lines))
print('\n'.join(lines[:9]))
