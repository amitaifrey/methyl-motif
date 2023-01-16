#awk '/^30$/{ print NR;}' dists > indices_30.txt


import pandas as pd
import numpy as np

idx = open('indices_30.txt', 'r')
l = []
for line in idx.readlines():
	l.append(int(line.strip()))
idx.close()

table = np.fromfile('/cs/staff/tommy/merged.lpairs', dtype=np.uint16).reshape(-1,4)
df = pd.DataFrame(table)
df = df.iloc[l]
df.to_csv('k30.csv')


import subprocess as sp
from tqdm import tqdm
import sqlite3
con = sqlite3.connect("hg19.db")
cur = con.cursor()
out = open('minus_seq10k.txt', 'w')
minus_idx = open('minus_30.txt', 'r')
for line in tqdm(minus_idx.readlines()):
	idx = int(line.strip()) - 1
	res = cur.execute(f"SELECT chr,end FROM hg19 where cpg='CpG{idx}'")
	chro, start = res.fetchone()
	start += 1
	res = cur.execute(f"SELECT start FROM hg19 where cpg='CpG{idx+1}'")
	end = res.fetchone()[0] - 1
	fasta_cmd = f"~Tommy/bin/hg19 {chro}:{start}-{end} +"
	ps2 = sp.Popen(fasta_cmd,shell=True,stdout=sp.PIPE,stderr=sp.STDOUT)
	output = ps2.communicate()[0].decode().upper().strip()
	out.write(output + '\n')
	out.flush()
out.close()

import subprocess as sp
from tqdm import tqdm
import sqlite3
con = sqlite3.connect("hg19.db")
cur = con.cursor()
out = open('plus_seq10k.txt', 'w')
plus_idx = open('plus_30.txt', 'r')
for line in tqdm(plus_idx.readlines()):
	idx = int(line.strip()) - 1
	res = cur.execute(f"SELECT chr,end FROM hg19 where cpg='CpG{idx}'")
	chro, start = res.fetchone()
	start += 1
	res = cur.execute(f"SELECT start FROM hg19 where cpg='CpG{idx+1}'")
	end = res.fetchone()[0] - 1
	fasta_cmd = f"~Tommy/bin/hg19 {chro}:{start}-{end} +"
	ps2 = sp.Popen(fasta_cmd,shell=True,stdout=sp.PIPE,stderr=sp.STDOUT)
	output = ps2.communicate()[0].decode().upper().strip()
	out.write(output + '\n')
	out.flush()
out.close()
