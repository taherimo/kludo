import atomium
import numpy as np
from igraph import Graph
from sklearn.metrics import pairwise_distances
import scipy.ndimage.measurements as ms

# import locale
#
# locale.setlocale(locale.LC_ALL, "")


def parse_pdb(pdb_file_path, chainid):

    chain = atomium.open(pdb_file_path).model.chain(chainid)

    atoms = sorted(list(chain.atoms()), key=lambda x: x.id)

    if chain is None:
        return 'invalid chain'

    aminoacids = []
    aminoacid_resnums = []
    aminoacid_letters = []
    aminoacid_ca_coords = []

    for res in chain.residues():

        # is_het = np.any([atm._is_hetatm for atm in res.atoms()])

        # if not is_het:
        aminoacids.append(res)
        aminoacid_resnums.append(res.id.split('.')[1])
        aminoacid_letters.append(res.code)
        atom_locs = {atom.name: atom.location for atom in res.atoms()}
        if 'CA' in atom_locs:
            aminoacid_ca_coords.append(atom_locs['CA'])
        else:
            aminoacid_ca_coords.append(np.mean(list(atom_locs.values()),axis=0))


    return aminoacids, atoms, aminoacid_ca_coords, aminoacid_letters, aminoacid_resnums, chain.radius_of_gyration

def parse_dssp(input_handle, qchainid):
    # import pandas as pd
    # df = pd.DataFrame(columns=('resnum', 'acc', 'relacc'))
    import re
    dssp = dict()

    # input_handle = open(file, 'r')
    start = False
    for line_byte in input_handle:

        line = line_byte.decode()

        if re.search('#  RESIDUE', line):
            if line.split()[1]=='RESIDUE':
                start = True
                continue

        if (start):
            if line[13:15].strip() != '!' or line[13:15].strip() != '!*':
                num = int(line[0:5].strip())
                resnum = line[5:11].strip()
                chainid = line[11:12].strip()
                aa = line[12:14].strip().upper()
                struct = line[14:25].strip()
                bp1 = line[25:29].strip()
                bp2 = line[29:33].strip()
                betasheet = line[33:34].strip()
                acc = float(line[34:38].strip())
                h_nho1 = line[38:50].strip()
                h_ohn1 = line[50:61].strip()
                h_nho2 = line[61:72].strip()
                h_ohn2 = line[72:83].strip()
                tco = line[83:91].strip()
                kappa = line[91:97].strip()
                alpha = line[97:103].strip()
                phi = line[103:109].strip()
                psi = line[109:115].strip()
                xca = line[115:122].strip()
                yca = line[122:129].strip()
                zca = line[129:136].strip()
                if aa != '!' and chainid == qchainid:
                    dssp[int(num)] = {"resnum": resnum, "struct": struct, "aa": aa, "acc": acc, "bp1": bp1, "bp2": bp2,
                                      "betasheet": betasheet, "h_nho1": h_nho1, "h_nho2": h_nho2, "h_ohn1": h_ohn1,
                                      "h_ohn2": h_ohn2}

    return dssp


def get_sec_struc_info(dssp, aminoacid_resnums):
    sec_struc_labels = ['-'] * len(aminoacid_resnums)
    betasheet_labels = ['-'] * len(aminoacid_resnums)
    beta_bridge_indices1 = ['-'] * len(aminoacid_resnums)
    beta_bridge_indices2 = ['-'] * len(aminoacid_resnums)

    for key in dssp:
        if dssp[key]['resnum'] in aminoacid_resnums:
            if dssp[key]['struct'][0:1] != '':
                index = aminoacid_resnums.index(dssp[key]['resnum'])
                sec_struc_labels[index] = dssp[key]['struct'][0:1]

                if dssp[key]['struct'][0:1] == 'E':
                    betasheet_labels[index] = dssp[key]['betasheet']

                    bp1_dssp_index = int(dssp[key]['bp1'])
                    bp2_dssp_index = int(dssp[key]['bp2'])

                    if bp1_dssp_index != 0 and bp1_dssp_index in dssp:
                        if dssp[bp1_dssp_index]['resnum'] in aminoacid_resnums:
                            if dssp[bp1_dssp_index]['betasheet'] == dssp[key]['betasheet']:
                                bp1_pdb_index = aminoacid_resnums.index(dssp[bp1_dssp_index]['resnum'])
                                beta_bridge_indices1[index] = bp1_pdb_index

                                # if beta_bridge_matrix[index, bp1_index] == beta_bridge_matrix[bp1_index, index] == '-':
                                #     beta_bridge_matrix[index, bp1_index] = beta_bridge_matrix[bp1_index, index] = dssp[key]['betasheet']

                    if bp2_dssp_index != 0 and bp2_dssp_index in dssp:
                        if dssp[bp2_dssp_index]['resnum'] in aminoacid_resnums:
                            if dssp[bp2_dssp_index]['betasheet'] == dssp[key]['betasheet']:
                                bp2_pdb_index = aminoacid_resnums.index(dssp[bp2_dssp_index]['resnum'])
                                beta_bridge_indices2[index] = bp2_pdb_index

                                # if beta_bridge_matrix[index, bp2_index] == beta_bridge_matrix[bp2_index, index] == '-':
                                #     beta_bridge_matrix[index, bp2_index] = beta_bridge_matrix[bp2_index, index] = dssp[key]['betasheet']

    return sec_struc_labels, betasheet_labels, beta_bridge_indices1, beta_bridge_indices2


def get_alpha_helices(sec_struct_labels):
    alphahelices = []
    for i in range(0, len(sec_struct_labels)):
        if sec_struct_labels[i] == 'H':
            if 0 < i < len(sec_struct_labels) - 1:
                if sec_struct_labels[i - 1] != 'H':
                    start = i
                if sec_struct_labels[i + 1] != 'H':
                    end = i
                    alphahelices.append((start, end))
            elif i == 0:
                start = i
            elif i == len(sec_struct_labels) - 1:
                end = i
                alphahelices.append((start, end))

    return alphahelices


def get_beta_strands(sec_struc_labels, betasheet_labels):
    beta_strands = []

    for i in range(0 , len(sec_struc_labels)):
        if sec_struc_labels[i] == 'E':
            if 0 < i < len(sec_struc_labels) - 1:
                if sec_struc_labels[i - 1] != 'E':
                    start = i
                if sec_struc_labels[i + 1] != 'E':
                    end = i
                    beta_strands.append((start, end, betasheet_labels[i]))
            elif i == 0:
                start = i
            elif i == len(sec_struc_labels) - 1:
                end = i
                beta_strands.append((start, end, betasheet_labels[i]))

    return beta_strands

def get_beta_bridges(dssp, aminoacid_resnums):

    beta_bridge_matrix = np.zeros([len(aminoacid_resnums), len(aminoacid_resnums)])

    for num in dssp:

        if dssp[num]['resnum'] in aminoacid_resnums:
            bp1 = int(dssp[num]['bp1'])
            bp2 = int(dssp[num]['bp2'])

            if bp1!=0 and bp1 in dssp.keys():
                i = aminoacid_resnums.index(dssp[num]['resnum'])
                j = aminoacid_resnums.index(dssp[bp1]['resnum'])
                beta_bridge_matrix[i,j] = beta_bridge_matrix[j,i] = 1

            if bp2!=0 and bp2 in dssp.keys():
                i = aminoacid_resnums.index(dssp[num]['resnum'])
                j = aminoacid_resnums.index(dssp[bp2]['resnum'])
                beta_bridge_matrix[i,j] = beta_bridge_matrix[j,i] = 1

    return beta_bridge_matrix

def extract_hydrpgen_bonds(dssp, aminoacid_resnums, energy_cutoff):

    hbonds_nho = np.zeros([len(aminoacid_resnums), len(aminoacid_resnums)])
    hbonds_ohn = np.zeros([len(aminoacid_resnums), len(aminoacid_resnums)])


    for num in dssp:

        if dssp[num]['resnum'] in aminoacid_resnums:

            h_nho1 = dssp[num]['h_nho1'].split(',')
            h_nho1_energy = float(h_nho1[1])
            h_nho1_target = int(h_nho1[0])
            h_nho2 = dssp[num]['h_nho2'].split(',')
            h_nho2_energy = float(h_nho2[1])
            h_nho2_target = int(h_nho2[0])
            h_ohn1 = dssp[num]['h_ohn1'].split(',')
            h_ohn1_energy = float(h_ohn1[1])
            h_ohn1_target = int(h_ohn1[0])
            h_ohn2 = dssp[num]['h_ohn2'].split(',')
            h_ohn2_energy = float(h_ohn2[1])
            h_ohn2_target = int(h_ohn2[0])

            if h_nho1_target != 0 and h_nho1_energy <= energy_cutoff:
                index1 = aminoacid_resnums.index(dssp[num]['resnum'])
                num2 = h_nho1_target + num
                if num2 in dssp and dssp[num2]['resnum'] in aminoacid_resnums:
                    index2 = aminoacid_resnums.index(dssp[num2]['resnum'])
                    # hbonds_nho[index1, index2] = h_nho1_energy
                    hbonds_nho[index1, index2] = 1

            if h_nho2_target != 0 and h_nho2_energy <= energy_cutoff:
                index1 = aminoacid_resnums.index(dssp[num]['resnum'])
                num2 = h_nho2_target + num
                if num2 in dssp and dssp[num2]['resnum'] in aminoacid_resnums:
                    index2 = aminoacid_resnums.index(dssp[num2]['resnum'])
                    # hbonds_nho[index1, index2] = h_nho2_energy
                    hbonds_nho[index1, index2] = 1

            if h_ohn1_target != 0 and h_ohn1_energy <= energy_cutoff:
                index1 = aminoacid_resnums.index(dssp[num]['resnum'])
                num2 = h_ohn1_target + num
                if num2 in dssp and dssp[num2]['resnum'] in aminoacid_resnums:
                    index2 = aminoacid_resnums.index(dssp[num2]['resnum'])
                    # hbonds_ohn[index1, index2] = h_ohn1_energy
                    hbonds_ohn[index1, index2] = 1

            if h_ohn2_target != 0 and h_ohn2_energy <= energy_cutoff:
                index1 = aminoacid_resnums.index(dssp[num]['resnum'])
                num2 = h_ohn2_target + num
                if num2 in dssp and dssp[num2]['resnum'] in aminoacid_resnums:
                    index2 = aminoacid_resnums.index(dssp[num2]['resnum'])
                    # hbonds_ohn[index1, index2] = h_ohn2_energy
                    hbonds_ohn[index1, index2] = 1


    return hbonds_nho, hbonds_ohn


def make_graph(atoms):

    atom_locs = [atm.location for atm in atoms]
    atom_resnums = np.array([atm.het.id.split('.')[1] for atm in atoms])
    atom_atom_dist = pairwise_distances(atom_locs, n_jobs = -1)
    atom_atom_cont = (atom_atom_dist <= 4).astype(int) - np.identity(atom_atom_dist.shape[0])
    # atoms_ca_mask = [atm.name=='CA' for atm in atoms]

    # resnums = sorted(list(set(atom_resnums)))
    used = []
    resnums = [x for x in atom_resnums if x not in used and (used.append(x) or True)]
    del used

    res_bounds_mask = atom_resnums[1:] != atom_resnums[0:-1]
    res_bounds_mask = np.insert(res_bounds_mask, 0, True)
    res_bounds_idx = np.where(res_bounds_mask)[0]

    g = Graph()
    for i in range(0, len(resnums)):
        g.add_vertex(i)

    # resnums = groupby(atom_resnums)
    # if len(resnums) != len(chain.residues()):
    #     print('oooops')
    # for k, g in resnums:
    #     print(k,g)

    atom_atom_cont = np.triu(atom_atom_cont, 1)
    res_res_cont = np.add.reduceat(np.add.reduceat(atom_atom_cont, res_bounds_idx, axis=0), res_bounds_idx, axis=1)
    res_res_cont= np.triu(res_res_cont, 1)

    nonzero_idx = np.nonzero(res_res_cont)

    for i in range(len(nonzero_idx[0])):
        g.add_edge(nonzero_idx[0][i], nonzero_idx[1][i], weight=res_res_cont[nonzero_idx[0][i],nonzero_idx[1][i]])

    # for i in range(res_res_cont.shape[0]-1):
    #     for j in range(i + 1, res_res_cont.shape[1]):
    #         if res_res_cont[i,j]>0:
    #             g.add_edge(i, j, weight=res_res_cont[i,j])

    # g['maxstrength'] = len(atoms)

    return g