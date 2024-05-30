import argparse
import os
import os.path as osp
import random
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="generate k-mer")
    # NPInter2 RPI369 RPI2241 RPI3265 RPI4158 RPI7317
    parser.add_argument('--dataset', default="NPInter2", help='project name')
    parser.add_argument('--kRna', default=4,type=int, help='kRna of k-mer') # 4 3
    parser.add_argument('--kProtein', default=3,type=int, help='kProtein of k-mer') # 3 2
    return parser.parse_args()
def read_sequence_file(path):
    name_list = []
    sequence_list = []
    sequence_file_path = path
    sequence_file = open(sequence_file_path, mode='r')

    for line in sequence_file.readlines():
        if line[0].strip()=='':
            continue
        if line[0] == '>':
            sequence_name = line.strip()[1:]
            name_list.append(sequence_name)
        else:
            sequence_list.append(line.strip())
    sequence_file.close()
    return name_list, sequence_list
def output_rna_k_mer_file(path,name_list, sequence_list):
    c = itertools.product(['A', 'C', 'G', 'U'], repeat=args.kRna)
    k_mers = []
    for i in c:
        k_mers.append(''.join(i))
    kmer_file = open(f'{path}/kmer.txt', mode='w')
    kmer_frequency_file =  open(f'{path}/kmer_frequency.txt', mode='w')
    for i in range(len(name_list)):
        my_dict = {value: 0 for value in k_mers}
        name = name_list[i]
        k_mer = [sequence_list[i][j:j+args.kRna] for j in range(0,len(sequence_list[i])-args.kRna+1)]
        for i in k_mer:
            my_dict[i] += 1
        kmer_frequency_file.write(name+','+','.join([str(y) for x,y in my_dict.items()])+'\n')
        kmer_file.write('>' + name + '\n')
        kmer_file.write(','.join(k_mer)+'\n')
    kmer_file.close()
    kmer_frequency_file.close()
def change_protein_sequence_20_to_7(path,protein_name_list, protein_sequence_list):
    '''
    使用简化的氨基酸字母表
    :return:
    '''
    simple_protein_file = open(path, mode='w')
    for i in range(len(protein_sequence_list)):
        sequence_list = list(protein_sequence_list[i])
        for j in range(len(sequence_list)):
            if sequence_list[j] == 'A' or sequence_list[j] == 'G' or sequence_list[j] == 'V':
                sequence_list[j] = 'A'
            elif sequence_list[j] == 'I' or sequence_list[j] == 'L' or sequence_list[j] == 'F' or sequence_list[j] == 'P':
                sequence_list[j] = 'B'
            elif sequence_list[j] == 'Y' or sequence_list[j] == 'M' or sequence_list[j] == 'T' or sequence_list[j] == 'S':
                sequence_list[j] = 'C'
            elif sequence_list[j] == 'H' or sequence_list[j] == 'N' or sequence_list[j] == 'Q' or sequence_list[j] == 'W':
                sequence_list[j] = 'D'
            elif sequence_list[j] == 'R' or sequence_list[j] == 'K':
                sequence_list[j] = 'E'
            elif sequence_list[j] == 'D' or sequence_list[j] == 'E':
                sequence_list[j] = 'F'
            elif sequence_list[j] == 'C':
                sequence_list[j] = 'G'
            elif sequence_list[j] == 'X':
                temp = random.sample(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1)[0]
                sequence_list[j] = temp
            else:
                print(sequence_list[j])
                print('protein sequence error')
                raise Exception
        protein_sequence_list[i] = ''.join(sequence_list)
        simple_protein_file.write('>' + protein_name_list[i] + '\n')
        simple_protein_file.write(protein_sequence_list[i]+'\n')
    simple_protein_file.close()
    return protein_sequence_list
def output_protein_k_mer_file(path,name_list, sequence_list):
    c = itertools.product(['A', 'B', 'C', 'D', 'E', 'F', 'G'], repeat=args.kProtein)
    k_mers = []
    for i in c:
        k_mers.append(''.join(i))
    kmer_file = open(f'{path}/kmer.txt', mode='w')
    kmer_frequency_file =  open(f'{path}/kmer_frequency.txt', mode='w')
    for i in range(len(name_list)):
        my_dict = {value: 0 for value in k_mers}
        name = name_list[i]
        k_mer = [sequence_list[i][j:j+args.kProtein] for j in range(0,len(sequence_list[i])-args.kProtein+1)]
        for i in k_mer:
            my_dict[i] += 1
        kmer_frequency_file.write(name+','+','.join([str(y) for x,y in my_dict.items()])+'\n')
        kmer_file.write('>' + name + '\n')
        kmer_file.write(','.join(k_mer)+'\n')
    kmer_file.close()
    kmer_frequency_file.close()
if __name__ == "__main__":
    print('start generate sequence k-mer\n')
    args = parse_args()
    path = f'../data/{args.dataset}'
    source_data_path = f'{path}/source_data'
    #rna
    rna_sequence_path = f'{source_data_path}/ncRNA_sequence.fasta'
    rna_name_list, rna_sequence_list = read_sequence_file(path=rna_sequence_path)
    rna_kmer_path = f'{path}/k_mer/rna'
    if not osp.exists(rna_kmer_path):
        os.makedirs(rna_kmer_path)
    output_rna_k_mer_file(rna_kmer_path,rna_name_list, rna_sequence_list )
    #protein
    protein_sequence_path = f'{source_data_path}/protein_sequence.fasta'
    protein_name_list, protein_sequence_list = read_sequence_file(path=protein_sequence_path)
    protein_kmer_path = f'{path}/k_mer/protein'
    if not osp.exists(protein_kmer_path):
        os.makedirs(protein_kmer_path)
    protein_sequence_list=change_protein_sequence_20_to_7(f'{protein_kmer_path}/simple_seq.txt',protein_name_list, protein_sequence_list)
    output_protein_k_mer_file(protein_kmer_path,protein_name_list, protein_sequence_list)
    print('generate sequence k-mer end\n')