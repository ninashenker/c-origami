import requests, sys, json

assembly_name = sys.argv[1]
save_path = sys.argv[2]

if assembly_name in ['hg38', 'hg19']:
    name = 'homo_sapiens'
    chr_list = list(range(1,23))
elif assembly_name in ['mm9', 'mm10']:
    name = 'mus_musculus'
    chr_list = list(range(1,20))
else:
    raise Exception('Assembly name not included')
 
chr_list.extend(['X', 'Y'])
chr_dict = {}

for chr_i in chr_list:
    server = "https://rest.ensembl.org"
    ext = f"/info/assembly/{name}/{chr_i}?" 
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})

    if not r.ok:
        r.raise_for_status()
        sys.exit()

    decoded = r.json()
    chr_dict[str(chr_i)] = decoded['length']

with open(f'{save_path}/chr_length.json', 'w') as f:
    json.dump(chr_dict, f)
