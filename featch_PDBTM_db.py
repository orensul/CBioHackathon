from __future__ import print_function

import sys
import os
import argparse
import requests
import xml.etree.ElementTree as ET

url = 'http://pdbtm.enzim.hu/data/pdbtmall'




class Chain:
	def __init__(self, id, num_transmembrane_segments, type):
		self.id = id
		self.num_transmembrane_segments = num_transmembrane_segments
		self.type = type
		self.seq = ""
		self.regions = []


class Region:
	def __init__(self, seq_start, seq_end, pdb_start, pdb_end, type):
		self.seq_start = seq_start
		self.seq_end = seq_end
		self.pdb_start = pdb_start
		self.pdb_end = pdb_end
		self.type = type


def get_database(prefix='.'):
	if not prefix.endswith('/'):
		prefix += '/'
	print('Fetching database...', file=sys.stderr)
	r = requests.get(url, stream=True)
	print('Saving database...', file=sys.stderr)
	f = open('%s/pdbtmall' % prefix, 'w')
	for line in r.iter_lines():
		decoded_line = line.decode("utf-8")
		if line:
			f.write(decoded_line)
	r.close()
	f.close()


def build_database(fn, prefix):
	print('Unpacking database...', file=sys.stderr)
	f = open(fn)
	db = f.read()
	f.close()
	firstline = 1
	header = ''
	entries = []
	pdbids = []
	for l in db.split('\n'):
		if firstline:
			header += l
			firstline -= 1
			continue
		if 'PDBTM>' in l:
			continue
		if l.startswith('<?'):
			continue
		if l.startswith('<pdbtm'):
			a = l.find('ID=') + 4
			b = a + 4
			pdbids.append(l[a:b])
			entries.append(header)
		entries[-1] += '\n' + l
	if not prefix.endswith('/'):
		prefix += '/'
	if not os.path.isdir(prefix):
		os.mkdir(prefix)
	for entry in zip(pdbids, entries):
		f = open(prefix + entry[0] + '.xml', 'w')
		f.write(entry[1])
		f.close()


def read_chains(prefix='.'):
	chains = []
	if not prefix.endswith('/'):
		prefix += '/'
	tree = ET.parse('%s/pdbtmall' % prefix)
	root = tree.getroot()
	for chain in root.iter('{http://pdbtm.enzim.hu}CHAIN'):
		chain_obj = Chain(id=chain.attrib['CHAINID'], num_transmembrane_segments=chain.attrib['NUM_TM'],
						  type=chain.attrib['TYPE'])
		for child in chain:
			tag = child.tag.split("{http://pdbtm.enzim.hu}", 1)[1]
			if tag == 'SEQ':
				chain_obj.seq = child.text
			if tag == 'REGION' and child.attrib['type']:
				region_obj = Region(seq_start=child.attrib['seq_beg'], seq_end=child.attrib['seq_end'],
									pdb_start=child.attrib['pdb_beg'], pdb_end=child.attrib['pdb_end'],
									type=child.attrib['type'])
				chain_obj.regions.append(region_obj)
		chains.append(chain_obj)
	return chains


def get_alpha_helix_subsequences(chains):
	res = []
	for chain in chains:
		if chain.type == "alpha":
			seq = chain.seq
			start_seq_positions = []
			end_seq_positions = []
			for region in chain.regions:
				if region.type == "H":
					start_seq_positions.append(int(region.seq_start))
					end_seq_positions.append(int(region.seq_end))
			start_seq_positions.sort()
			end_seq_positions.sort()
			seq = seq.replace(" ", "")
			for i in range(len(start_seq_positions)):
				res.append(seq[start_seq_positions[i]-1:end_seq_positions[i]])
	return res






def main():
	parser = argparse.ArgumentParser(
		description='Manages PDBTM databases. Automatically fetches the PDBTM database if no options are '
					'specified. Run without any arguments, dbtool will retrieve the PDBTM database, '
					'store it in pdbtm, and unpack it.')
	parser.add_argument('-d', '--db', default='pdbtmall', help='name of db file')
	parser.add_argument('-b', '--build-db', action='store_true', help='rebuild database from an existing pdbtmsall file (available at http://pdbtm.enzim.hu/data/pdbtmall)')
	parser.add_argument('directory', nargs='?', default='pdbtm', help='directory to store database in')
	parser.add_argument('-f', '--force-refresh', action='store_true', help='overwrite of existing db')
	args = parser.parse_args()

	if args.build_db:
		build_database(args.db, args.directory)
	else:
		if not os.path.isdir(args.directory):
			os.mkdir(args.directory)
		if args.force_refresh or not os.path.isfile('%s/%s' % (args.directory, args.db)):
			get_database(args.directory)
		build_database('%s/%s' % (args.directory, args.db), args.directory)

	chains = read_chains(args.directory)
	alpha_helix_subsequences = get_alpha_helix_subsequences(chains)
	print(alpha_helix_subsequences)


if __name__ == '__main__':
	main()