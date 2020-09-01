import os, sys, itertools, math
import numpy as np
from igraph import Graph
from scipy.linalg import expm
import Bio.PDB
from collections import Counter
import ntpath
from sklearn.cluster import SpectralClustering
import subprocess as sp
from collections import defaultdict
import pickle
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from operator import itemgetter
from prettytable import PrettyTable,MSWORD_FRIENDLY,ALL
import warnings


here = os.path.dirname(__file__)

warnings.filterwarnings("ignore")

max_asa = dict()

max_asa['A'] = 113
max_asa['R'] = 241
max_asa['N'] = 158
max_asa['D'] = 151
max_asa['C'] = 140
max_asa['E'] = 183
max_asa['Q'] = 189
max_asa['G'] = 85
max_asa['H'] = 194
max_asa['I'] = 182
max_asa['L'] = 180
max_asa['K'] = 211
max_asa['M'] = 204
max_asa['F'] = 218
max_asa['P'] = 143
max_asa['S'] = 122
max_asa['T'] = 146
max_asa['W'] = 259
max_asa['Y'] = 229
max_asa['V'] = 160

# for line in open("data/max-asa.csv").readlines()[1:]:
#     line_split = line.split(',')
#     max_asa[line_split[1]] = float(line_split[4])

max_asa['B'] = (max_asa['N'] + max_asa['D']) / 2
max_asa['Z'] = (max_asa['Q'] + max_asa['E']) / 2
max_asa['J'] = (max_asa['I'] + max_asa['L']) / 2
max_asa['X'] = max_asa['O'] = max_asa['U'] = sum(max_asa.values()) / len(max_asa)


hydrophobicity = dict()

hydrophobicity['A'] = 1.8
hydrophobicity['C'] = 2.5
hydrophobicity['U'] = 2.5
hydrophobicity['D'] = -3.5
hydrophobicity['B'] = -3.5
hydrophobicity['E'] = -3.5
hydrophobicity['F'] = 2.8
hydrophobicity['G'] = -0.4
hydrophobicity['H'] = -3.2
hydrophobicity['I'] = 4.5
hydrophobicity['K'] = -3.9
hydrophobicity['O'] = -3.9
hydrophobicity['L'] = 3.8
hydrophobicity['M'] = 1.9
hydrophobicity['N'] = -3.5
hydrophobicity['P'] = -1.6
hydrophobicity['Q'] = -3.5
hydrophobicity['R'] = -4.5
hydrophobicity['S'] = -0.8
hydrophobicity['T'] = -0.7
hydrophobicity['V'] = 4.2
hydrophobicity['W'] = -0.9
hydrophobicity['Y'] = -1.3
hydrophobicity['X'] = -0.5
hydrophobicity['Z'] = -3.5


def make_graph(aminoacids,aminoacid_ca_coords, co_alpha_helix_matrix, co_beta_strand_matrix,betasheet_labels, acc, hydphob,hydrogen_bonds,beta_bridges):

    loaded_model = pickle.load(open(os.path.join(here, 'edge-weight-predictor.sav'), 'rb'))

    g = Graph()
    main_chain_atoms = {'N', 'CA', 'C', 'O'}
    for i in range(0, len(aminoacids)):
        g.add_vertex([i],acc = acc[i], hydphob= hydphob[i], ca_coord = aminoacid_ca_coords[i])

    for i in range(0, len(aminoacids) - 1):

        for j in range(i + 1, len(aminoacids)):



            ca_dist = np.linalg.norm(aminoacid_ca_coords[i] - aminoacid_ca_coords[j])
            if ca_dist <= 15:
                # g.add_edge(i, j, weight=gamma * co_beta_strand_matrix[i,j] + 1)

                num_all_contacts = 0
                num_bb_contacts = 0
                for atom1 in aminoacids[i]:
                    if atom1.get_name() in main_chain_atoms:
                        for atom2 in aminoacids[j]:
                            dist = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                            if dist <= 4:
                                num_all_contacts += 1
                                if atom2.get_name() in main_chain_atoms:
                                    num_bb_contacts += 1
                    else:
                        for atom2 in aminoacids[j]:
                            dist = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                            if dist <= 4:
                                num_all_contacts += 1

                if num_all_contacts > 0:
                    mean_relacc = (acc[i] + acc[j]) / 2
                    mean_hphob = (hydphob[i] + hydphob[j]) / 2
                    in_same_betasheet = 0
                    if betasheet_labels[i] == betasheet_labels[j] != '-':
                        in_same_betasheet = 1
                    beta_bridge_in_same_beta_sheet = 0
                    if beta_bridges[i, j] != 0 and in_same_betasheet == 1:
                        beta_bridge_in_same_beta_sheet = 1


                    result = loaded_model.predict_proba(np.array([num_bb_contacts,num_all_contacts ,co_beta_strand_matrix[i,j],co_alpha_helix_matrix[i,j],hydrogen_bonds[i,j],beta_bridges[i,j],in_same_betasheet,beta_bridge_in_same_beta_sheet,mean_relacc,mean_hphob,ca_dist,j-i]).reshape(1, -1))[0]


                    g.add_edge(i, j, weight = result[1])


    return g


def parse_pdb(pdb_file_path, pdbid, chainid):
    pdbparser = Bio.PDB.PDBParser()
    structure = pdbparser.get_structure(pdbid, pdb_file_path)
    model = structure[0]

    if chainid not in model:
        return 'invalid chain'

    chain = model[chainid]


    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
         'SEC': 'U', 'PYL': 'O', 'GLX': 'Z', 'ASX': 'B', 'CGU': 'X',
         'MSE': 'X', 'BHD': 'X', 'CSS': 'X', 'PCA': 'X', 'TPQ': 'X',
         'FME': 'X', 'OCS': 'X', 'ABA': 'X', 'SEP': 'X', 'HIC': 'X',
         'TRQ': 'X', 'M3L': 'X', 'CSO': 'X', 'TPO': 'X', 'MHO': 'X',
         'BH2': 'X', 'UNK': 'X', 'MEN': 'X', 'LLP': 'X', 'HYP': 'X',
         'TRN': 'X', 'FGL': 'X', 'CSB': 'X', 'MDO': 'X', 'SVA': 'X',
         'GMA': 'X', 'SME': 'X', 'PTR': 'X', 'CSD': 'X', 'KCX': 'X',
         'CYG': 'X', 'CCS': 'X', 'SCH': 'X', 'TRO': 'X', 'SE7': 'X',
         'NLE': 'X', 'ORN': 'X', 'MHS': 'X', 'AGM': 'X', 'MGN': 'X',
         'SMC': 'X', 'CSX': 'X', 'LYZ': 'X', 'EYS': 'X', 'CSP': 'X',
         'CXM': 'X', 'MLZ': 'X', 'AAR': 'X', 'MEA': 'X', 'NEP': 'X',
         'MIS': 'X', 'CME': 'X', 'SAC': 'X', 'CYN': 'X', 'DNP': 'X',
         'BTN': 'X', 'IAS': 'X', 'MSO': 'X', 'ALS': 'X', 'CAS': 'X',
         'DDZ': 'X', 'MLY': 'X', 'CGN': 'X', 'BFD': 'X', 'CZZ': 'X',
         'CMH': 'X', 'MME': 'X', '143': 'X', 'PHD': 'X', 'AR4': 'X',
         'DSN': 'X', 'YCM': 'X', 'TYI': 'X', '175': 'X', 'NCB': 'X',
         'NC1': 'X', 'CSR': 'X', 'AHB': 'X', 'AYA': 'X', 'LAL': 'X',
         'TRW': 'X', 'MCL': 'X', 'TYS': 'X', 'TYY': 'X', 'SEB': 'X',
         'LED': 'X', 'KST': 'X', 'LCK': 'X', 'DMG': 'X', 'OHI': 'X',
         'CS4': 'X', 'CSU': 'X', 'OMT': 'X', 'PR3': 'X', 'MED': 'X',
         'SNC': 'X', 'PSH': 'X', 'NIY': 'X', '5VV': 'X', 'DTY': 'X',
         'SLZ': 'X', 'ALY': 'X', 'ESD': 'X', 'FTR': 'X', 'LLZ': 'X',
         'KPI': 'X', 'DAB': 'X', '0AF': 'X', 'BIF': 'X', 'XCN': 'X',
         'HOX': 'X', 'NFA': 'X', 'ARO': 'X', 'CSZ': 'X', 'HIP': 'X',
         'QPA': 'X', 'N80': 'X', 'M0H': 'X', 'AGT': 'X', 'PN2': 'X',
         '4HH': 'X', 'HTR': 'X', 'TOX': 'X', 'PBF': 'X', 'OLD': 'X',
         'KFP': 'X', '4IN': 'X', 'DHA': 'X', 'OSE': 'X', 'MIR': 'X',
         'FP9': 'X', 'FT6': 'X', '2CO': 'X', 'SDP': 'X', 'OYL': 'X',
         'SCY': 'X', 'SCS': 'X', '6DN': 'X', 'HAR': 'X', 'LYR': 'X',
         'CAF': 'X', 'HIQ': 'X', '0A9': 'X', 'BCS': 'X', 'BYR': 'X',
         'DYA': 'X', 'DDE': 'X', 'SL5': 'X', 'CSW': 'X', 'CGV': 'X',
         'SEE': 'X', 'DAL': 'X', 'OCY': 'X', 'IYR': 'X', 'KOR': 'X',
         '6V1': 'X', 'CY3': 'X'}

    aminoacids = []
    aminoacid_resnums = []
    aminoacid_letters = []
    aminoacid_ca_coords = []

    for res in Bio.PDB.Selection.unfold_entities(chain, 'R'):
        # if res.get_id()[0] == ' ' and res.get_resname() != 'UNK':
        # print(res.get_id(), res.get_resname())
        # if res.get_resname() not in {'HOH','WAT','UNK',' CA', 'BOG', 'CMP','FES',' CO','SO4',' ZN',' MG','NAD','BGC','FUC'}:
        if res.get_resname() in d.keys():
            resnum = str(res.get_id()[1])
            if res.get_id()[2] != ' ':
                resnum = resnum + res.get_id()[2]
            aminoacid_resnums.append(resnum)
            aminoacid_letters.append(d[res.get_resname()])

            if 'CA' in res:
                aminoacid_ca_coords.append(res['CA'].get_coord())
            else:
                atom_coords = []
                for atom in res:
                    atom_coords.append(atom.get_coord())
                atom_coords_sum = [sum(x) for x in zip(*atom_coords)]
                aminoacid_ca_coords.append(np.asarray([x / len(atom_coords) for x in atom_coords_sum]))

            aminoacids.append(res)

    return aminoacids, aminoacid_ca_coords, aminoacid_letters, aminoacid_resnums


def parse_dssp(input_handle, qchainid):
    # import pandas as pd
    # df = pd.DataFrame(columns=('resnum', 'acc', 'relacc'))
    import re
    dssp = dict()

    # input_handle = open(file, 'r')
    start = False
    for line_byte in input_handle:

        line = line_byte.decode()

        if re.search('#', line):
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


def merge_lists(x): return sum(x, [])


def contract_alpha_helices(graph, alpha_helices, max_length):

    n = len(graph.vs)

    assignments_for_contraction = [None] * n

    selected_alpha_helices = []
    for alpha_helix in alpha_helices:
        if alpha_helix[1] - alpha_helix[0] + 1 <= max_length:
            selected_alpha_helices.append(alpha_helix)

    counter = len(selected_alpha_helices)

    for i in range(0, n):
        in_alpha_helix = False
        for j in range(0, len(selected_alpha_helices)):
            if selected_alpha_helices[j][0] <= graph.vs[i]['name'][0] <= selected_alpha_helices[j][1]:
                assignments_for_contraction[i] = j
                in_alpha_helix = True
        if not in_alpha_helix:
            assignments_for_contraction[i] = counter
            counter += 1

    graph.contract_vertices(assignments_for_contraction, combine_attrs=dict(name=merge_lists,acc='mean', hydphob = 'mean'))

    graph.simplify(combine_edges=dict(weight=sum))


def get_segments(assignment,query=None):

    start = 0
    segments = []
    segment_labels = []

    for i in range(1, len(assignment) - 1):
        if assignment[i] != assignment[i - 1]:
            if query is not None:
                if assignment[i - 1] in query:
                    end = i - 1
                    segments.append((start, end))
                    segment_labels.append(assignment[end])
                if assignment[i] in query:
                    start = i
            else:
                end = i - 1
                segments.append((start, end))
                segment_labels.append(assignment[end])
                start = i

    if assignment[-1] != assignment[-2]:
        end = len(assignment) - 2
        segments.append((start, end))
        segment_labels.append(assignment[end])
        start = end = len(assignment) - 1
    else:
        end = len(assignment) - 1

    segments.append((start, end))
    segment_labels.append(assignment[end])

    return segments, segment_labels


def get_shortest_segment_index(segments,segment_labels):

    segment_count = Counter(segment_labels)
    multi_segment_domains = {key:value for key,value in segment_count.items() if value > 1}
    segment_ids_filtered = [x for x in range(len(segments)) if segment_labels[x] in multi_segment_domains]
    # segment_ids_filtered = [int(x) for x in segment_count_filtered.keys()]

    # if len(segment_ids_filtered) > 0:

    # shortest_segment_index = segment_ids_filtered[0]
    shortest_segment_index = 0
    min_length = segments[shortest_segment_index][1] - segments[shortest_segment_index][0] + 1


    # for segment_id in segment_ids_filtered[1:]:
    for segment_id in range(1,len(segments)):
        length = segments[segment_id][1] - segments[segment_id][0] + 1

        if length < min_length:
            shortest_segment_index = segment_id
            min_length = length

    return shortest_segment_index


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


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # return array[idx]
    return idx


def get_domains_as_segments(assignment):

    start = 0
    segments = defaultdict(list)

    for i in range(1, len(assignment)):
        if assignment[i] != assignment[i - 1]:
            end = i - 1
            segments[assignment[i-1]].append((start, end))
            start = i

    end = len(assignment) - 1
    segments[assignment[len(assignment) - 1]].append((start, end))

    return segments

def get_domains_as_segments_by_resnum(domains_as_segments, aminoacid_resnums):

    domains_as_segments_by_aa_num = defaultdict(list)

    for key in domains_as_segments:
        for segment in domains_as_segments[key]:
            domains_as_segments_by_aa_num[key].append((aminoacid_resnums[segment[0]], aminoacid_resnums[segment[1]]))

    return domains_as_segments_by_aa_num


def conv_to_text(domains,delimiter='\t'):
    # standard_format = str(len(domains))
    standard_format = ''
    for i in domains:
        # if i > 1:
        #     standard_format += '\t'
        if standard_format == '':
            standard_format ='('
        else:
            standard_format += delimiter+ '('
        # standard_format += pdbid + chainid + str(i) + ': '
        for j in range(0, len(domains[i])):
            standard_format += str(domains[i][j][0]) + '-' + str(domains[i][j][1])
            if j < len(domains[i]) - 1:
                standard_format += ','
        standard_format += ')'

    return standard_format



def remove_short_segments(assignment,cutoff, distance_matrix = None):

    segments,segment_labels = get_segments(assignment)
    shortest_segment_index = get_shortest_segment_index(segments,segment_labels)
    if shortest_segment_index != None:
        shortest_segment = segments[shortest_segment_index]
        shortest_segment_length = shortest_segment[1] - shortest_segment[0] + 1
    else:
        shortest_segment_length = cutoff + 1000

    while shortest_segment_length < cutoff:
        if 0 < shortest_segment_index < len(segments) - 1:
            pred_assignment = assignment[segments[shortest_segment_index - 1][0]]
            succ_assignment = assignment[segments[shortest_segment_index + 1][0]]

            if pred_assignment == succ_assignment:
                for j in range(shortest_segment[0], shortest_segment[1] + 1):
                    assignment[j] = pred_assignment
            elif distance_matrix is not None:
                # current_ca_coords = [aminoacid_coords[j] for j in range(segments[shortest_segment_index][0], segments[shortest_segment_index][1] + 1)]
                # pred_ca_coords = [aminoacid_coords[j] for j in range(segments[shortest_segment_index - 1][0], segments[shortest_segment_index - 1][1] + 1)]
                # succ_ca_coords = [aminoacid_coords[j] for j in range(segments[shortest_segment_index + 1][0], segments[shortest_segment_index + 1][1] + 1)]

                current_segment_res_ids = range(segments[shortest_segment_index][0], segments[shortest_segment_index][1] + 1)
                pred_segment_res_ids = range(segments[shortest_segment_index - 1][0], segments[shortest_segment_index - 1][1] + 1)
                succ_segment_res_ids = range(segments[shortest_segment_index + 1][0], segments[shortest_segment_index + 1][1] + 1)

                # dist_with_pred = average_linkage_distance(current_ca_coords, pred_ca_coords)
                # dist_with_succ = average_linkage_distance(current_ca_coords, succ_ca_coords)

                dist_with_pred = 0
                dist_with_succ = 0
                for i in current_segment_res_ids:
                    for j in pred_segment_res_ids:
                        dist_with_pred += distance_matrix[i,j]
                    for j in succ_segment_res_ids:
                        dist_with_succ += distance_matrix[i,j]

                dist_with_pred /= len(current_segment_res_ids) * len(pred_segment_res_ids)
                dist_with_succ /= len(current_segment_res_ids) * len(succ_segment_res_ids)


                if dist_with_pred < dist_with_succ:
                    for j in range(shortest_segment[0], shortest_segment[1] + 1):
                        assignment[j] = pred_assignment
                else:
                    for j in range(shortest_segment[0], shortest_segment[1] + 1):
                        assignment[j] = succ_assignment

        elif shortest_segment_index == 0:
            succ_assignment = assignment[segments[shortest_segment_index + 1][0]]
            for j in range(shortest_segment[0], shortest_segment[1] + 1):
                assignment[j] = succ_assignment
        elif shortest_segment_index == len(segments) - 1:
            pred_assignment = assignment[segments[shortest_segment_index - 1][0]]
            for j in range(shortest_segment[0], shortest_segment[1] + 1):
                assignment[j] = pred_assignment

        segments,segment_labels = get_segments(assignment)
        shortest_segment_index = get_shortest_segment_index(segments,segment_labels)
        if shortest_segment_index != None:
            shortest_segment = segments[shortest_segment_index]
            shortest_segment_length = shortest_segment[1] - shortest_segment[0] + 1
        else:
            shortest_segment_length = cutoff + 1000


def average_linkage_distance(coords1, coords2):
    total_dist = 0
    for coord1 in coords1:
        for coord2 in coords2:
            dist = np.linalg.norm(coord1 - coord2)
            total_dist += dist

    return total_dist / (len(coords1) * len(coords2))


def reassign_short_segments(graph,segments,num_domains, labels,aminoacid_resnums,min_seg_size):
    min_between_domain_weight = float('inf')
    small_segments_idx = get_small_segments_idx(segments, min_seg_size)
    all_partiotionings = list(itertools.product(range(num_domains), repeat=len(small_segments_idx)))
    for partitioning in all_partiotionings:
        labels_temp = labels.copy()

        for i in range(len(small_segments_idx)):
            segment = segments[small_segments_idx[i]]
            # print('segment ' + str(small_segments_idx[i]) + ' in domain ' + str(x[i]))
            labels_temp[segment[0]:segment[1]+1] = partitioning[i]

        segments_temp, _ = get_segments(labels_temp)
        small_segments_idx_temp = get_small_segments_idx(segments_temp, min_seg_size)
        # if len(small_segments_idx_temp) == 0 and len(set(labels_temp))==num_domains:
        if len(small_segments_idx_temp) == 0:
            clusters_by_resnum_temp, clusters_by_vtx_index_temp, clusters_by_index_temp = get_clusters_by_aminoacid_labels(graph, aminoacid_resnums, labels_temp)
            between_domain_weight = 0
            for i in range(len(clusters_by_vtx_index_temp)-1):
                for j in range(i+1, len(clusters_by_vtx_index_temp)):
                    for ii in clusters_by_vtx_index_temp[i]:
                        for jj in clusters_by_vtx_index_temp[j]:
                            e_id = graph.get_eid(ii, jj, directed=False,error=False)
                            if e_id != -1:
                                between_domain_weight += graph.es[e_id]['weight']

            if min_between_domain_weight > between_domain_weight:
                min_between_domain_weight = between_domain_weight
                for i in range(len(small_segments_idx)):
                    segment = segments[small_segments_idx[i]]
                    # print('segment ' + str(small_segments_idx[i]) + ' in domain ' + str(x[i]))
                    labels[segment[0]:segment[1] + 1] = partitioning[i]



def remove_redundant_segments(labels, num_domains, seg_numdomians_ratio, distance_matrix):
    segments, segment_labels = get_segments(labels)
    shortest_segment_index = get_shortest_segment_index(segments, segment_labels)
    if shortest_segment_index != None:
        shortest_segment = segments[shortest_segment_index]
    else:
        return None

    while len(segments) / float(num_domains) > seg_numdomians_ratio:
        remove_short_segments(labels,shortest_segment[1] - shortest_segment[0] + 2, distance_matrix)
        segments, segment_labels = get_segments(labels)
        shortest_segment_index = get_shortest_segment_index(segments, segment_labels)
        if shortest_segment_index is not None:
            shortest_segment = segments[shortest_segment_index]
        else:
            break


def calc_sil(clusters, distance_matrix):

    n = distance_matrix.shape[0]

    a = np.zeros(n)
    b = np.zeros(n)

    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i != j:
                for k in clusters[i]:
                    # total = 0
                    distances= sorted(distance_matrix[k,clusters[j]])
                    total = sum(distances) / len(distances)
                    # for p in clusters[j]:
                    #     total += kernel_matrix[k,p]
                    # total /= len(clusters[j])
                    # print(k, i, j, total)
                    b[k] = min(b[k], total)
            else:
                for k in clusters[i]:
                    # total = 0
                    distances = []
                    for p in clusters[j]:
                        if k != p:
                            # total += kernel_matrix[k,p]
                            distances.append(distance_matrix[k,p])
                            distances.sort()
                    total = sum(distances) / len(distances)
                    # total /= len(clusters[j]) - 1
                    a[k] = total

    sil_scores = [(x2 - x1)/ max(x1,x2) for (x1, x2) in zip(a, b)]
    return np.mean(sil_scores)


def get_small_segments_idx(segments, min_seg_size):

    segments_size = [segments[i][1] - segments[i][0] + 1 for i in range(len(segments))]

    small_segments_idx = []
    for i in range(len(segments_size)):
        if segments_size[i] < min_seg_size:
            small_segments_idx.append(i)

    return small_segments_idx


def get_clusters_by_vtx_labels(graph, aminoacid_resnums, labels):
    clusters_by_resnum = []
    clusters_by_vtx_index = []
    clusters_by_index = []

    for k in set(labels):
        # for k in set(communities._membership):
        cluster_by_vtx_index = [i for i, x in enumerate(labels) if x == k]
        cluster_by_index = sum([graph.vs[i]['name'] for i in cluster_by_vtx_index], [])
        cluster_by_resnum = [aminoacid_resnums[i] for i in cluster_by_index]
        clusters_by_vtx_index.append(cluster_by_vtx_index)
        clusters_by_index.append(cluster_by_index)
        clusters_by_resnum.append(cluster_by_resnum)

    return clusters_by_resnum, clusters_by_vtx_index, clusters_by_index

def get_clusters_by_aminoacid_labels(graph, aminoacid_resnums, labels):
    clusters_by_resnum = []
    clusters_by_vtx_index = []
    clusters_by_index = []

    for k in set(labels):
        # for k in set(communities._membership):
        cluster_by_index = [i for i, x in enumerate(labels) if x == k]
        cluster_by_vtx_index = []
        for i in cluster_by_index:
            for j in range(len(graph.vs)):
                if i in graph.vs[j]['name']:
                    if j not in cluster_by_vtx_index:
                        cluster_by_vtx_index.append(j)

        cluster_by_resnum = [aminoacid_resnums[i] for i in cluster_by_index]
        clusters_by_vtx_index.append(cluster_by_vtx_index)
        clusters_by_index.append(cluster_by_index)
        clusters_by_resnum.append(cluster_by_resnum)

    return clusters_by_resnum, clusters_by_vtx_index, clusters_by_index


def convert_kernel_to_distance(kernel_matrix, method):
    n = kernel_matrix.shape[0]
    distance_matrix = np.zeros((n, n))

    if method == 'norm':
        for i in range(n - 1):
            for j in range(i + 1, n):
                x = kernel_matrix[i,i] - 2 * kernel_matrix[i,j] + kernel_matrix[j,j]
                if x < 0:
                    x = 0
                distance_matrix[i,j] = distance_matrix[j,i] = math.sqrt(x)

        return distance_matrix

    elif method == 'CS':
        for i in range(n - 1):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = math.acos(kernel_matrix[i, j] ** 2) / (kernel_matrix[i,i] * kernel_matrix[j,j])

        return distance_matrix

    elif method == 'exp':
        for i in range(n - 1):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = math.exp(-1 * kernel_matrix[i, j])

        return distance_matrix

    elif method == 'naive':
        for i in range(n - 1):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = 1 - (kernel_matrix[i,j] / math.sqrt(kernel_matrix[i,i] * kernel_matrix[j,j]))

        return distance_matrix



def cluster(num_domains, main_graph,graph, aminoacid_resnums, diff_kernel, min_seg_size, seg_numdomians_ratio,distance_matrix,min_domain_size):


    clustering = SpectralClustering(n_clusters=num_domains, assign_labels="kmeans", random_state=0,affinity='precomputed', n_init=100).fit(diff_kernel)

    clusters_by_resnum, clusters_by_vtx_index, clusters_by_index = get_clusters_by_vtx_labels(graph,aminoacid_resnums,clustering.labels_)

    labels = np.zeros(len(aminoacid_resnums),dtype=int)

    # labels = [None] * len(aminoacid_resnums)

    for i in range(len(clusters_by_index)):
        labels[clusters_by_index[i]] = i

    labels_by_vertices = np.zeros(len(graph.vs), dtype=int)

    for i in range(len(clusters_by_vtx_index)):
        labels_by_vertices[clusters_by_vtx_index[i]]= i


    remove_short_segments(labels, min_seg_size, distance_matrix)

    remove_redundant_segments(labels,num_domains,seg_numdomians_ratio,distance_matrix)


    if(len(set(labels)) < num_domains):
        return 'error'

    hydphob = main_graph.vs['hydphob']

    # hydphob_acc_ratio = [(hydphob[i] + max(hydphob)) / ((acc[i] + 0.00001) * 2 * max(hydphob)) for i in
    #                      range(len(hydphob))]

    # core_res_idx = [idx for idx, val in enumerate(hydphob_acc_ratio) if val > 0]

    core_res_idx = [i for i in range(len(hydphob)) if hydphob[i] > 3]


    labels_core = [labels[i] for i in core_res_idx]

    if len(set(labels_core)) <= 1:
        return 'error'

    if not (2 <= len(set(labels_core)) <= len(labels_core) - 1):
        return 'error'

    core_dist_matrix = distance_matrix[core_res_idx,:][:,core_res_idx]

    # sil_score= silhouette_score(distance_matrix, labels=labels)
    sil_score = silhouette_score(core_dist_matrix, labels=labels_core)

    for label in set(labels):
        if np.count_nonzero(labels == label) < min_domain_size:
            return 'error'

    return labels,labels_by_vertices, sil_score


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

def reg_lap_kernel(graph, alpha): # aka normalized random walk with restart kernel

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    lap_matrix = deg_matrix - adj_matrix

    kernel = np.linalg.inv(np.identity(np.shape(lap_matrix)[0]) + alpha * lap_matrix)


    return kernel

def markov_exp_diff_kernel(graph, beta):

    num_vtx = len(graph.vs)

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    m_matrix = (deg_matrix - adj_matrix - num_vtx * np.identity(np.shape(adj_matrix)[0]))/num_vtx

    kernel = expm(-1 * beta * m_matrix)


    while np.isinf(kernel).any():
        beta -= beta / 100
        # beta -= 1
        kernel = expm(-1 * beta * m_matrix)
    # print(beta)

    return kernel

def markov_diff_kernel(graph, t):

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    p_matrix = np.zeros(shape=(np.shape(adj_matrix)[0],np.shape(adj_matrix)[1]))

    for i in range (np.shape(adj_matrix)[0]):
        for j in range (np.shape(adj_matrix)[1]):
            if deg_matrix[i,i] > 0:
                p_matrix[i,j]= adj_matrix[i,j] / deg_matrix[i,i]
            elif deg_matrix[i,i] == 0:
                p_matrix[i, j] = adj_matrix[i, j] / 0.00001

    z_matrix = np.zeros(shape=(np.shape(adj_matrix)[0],np.shape(adj_matrix)[0]))

    for tau in np.arange(1, int(t + 1), step=1):
        z_matrix = z_matrix + np.linalg.matrix_power(p_matrix, tau)
    z_matrix = z_matrix / t

    kernel = np.matmul(z_matrix, z_matrix.transpose())


    return kernel

def lap_exp_diff_kernel(graph, beta):

    adj_matrix = np.array(graph.get_adjacency(attribute='weight').data)
    deg_matrix = np.diag(graph.strength(weights='weight'))
    lap_matrix = deg_matrix - adj_matrix

    kernel = expm(-1 * beta * lap_matrix)

    return kernel


def exponential(x, a, b):
    return a*np.exp(x*b)


def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
      a = num[:-2-(not dec)]       # integer part
      b = int(num[-2-(not dec)])+1 # decimal part
      return float(a)+b**(-dec+1) if a and b == 10 else float(a+str(b))
    return float(num[:-1])


help_text = """

  -help                     Help
  -pdb [PATH]               PDB file Path (*)
  -chainid [ID]             Chain id (*)
  -dssppath [PATH]          DSSP binary file path (*)
  -numdomains [NUMBER]      The number of domains
  -minsegsize [SIZE]        Minimum segment size
  -mindomainsize [SIZE]     Minimum domain size
  -maxalphahelix [SIZE]     Maximum size of alpha-helix to contract
  -maxsegdomratio [RATIO]   Maximum ratio of segment count to domain count
  -kernel [TYPE]            The type of graph node kernel (**)
  -dispall                  Display all candidate partitionings
  -diffparamx [VALUE]       Diffusion parameter X (***)
  -diffparamy [VALUE]       Diffusion parameter Y (***)
 
 *   These arguments are necessary
 
 **  Type should be choosen from: lap-exp-diff, markov-diff, reg-lap-diff and markov-exp-diff

 *** The parameters diffparamx and diffparamy are coefficient (x) and exponent (y) of node count (n),
     respectively, which determine the diffusion parameter, t, for each kernel. (t=xn^y)
"""

def run(argv):

    pdb_file_path = ''
    chain_id = ''
    num_domains = None
    min_seg_size = 25
    max_alpha_helix_size_to_contract = 30
    seg_numdomians_ratio = 1.6
    min_domain_size = 27
    kernel = 'lap-exp-diff'
    display_all_partiotionings = False
    diff_param_x = None
    diff_param_y = None
    dssp_path = ''

    argument_error = False

    for i in range(0, len(argv)):
        if argv[i][0] == '-':
            if argv[i] == '-help':
                print(help_text)
                return 'help'
            elif argv[i] == '-pdb':
                pdb_file_path = argv[i + 1]
            elif argv[i] == '-chainid':
                chain_id = argv[i + 1].upper()
            elif argv[i] == '-numdomains':
                num_domains = int(argv[i + 1])
            elif argv[i] == '-minsegsize':
                min_seg_size = int(argv[i + 1])
            elif argv[i] == '-mindomainsize':
                min_domain_size = int(argv[i + 1])
            elif argv[i] == '-maxalphahelix':
                max_alpha_helix_size_to_contract = int(argv[i + 1])
            elif argv[i] == '-maxsegdomratio':
                seg_numdomians_ratio = float(argv[i + 1])
            elif argv[i] == '-kernel':
                kernel = argv[i + 1]
            elif argv[i] == '-dispall':
                display_all_partiotionings = True
            elif argv[i] == '-diffparamx':
                diff_param_x = float(argv[i + 1])
            elif argv[i] == '-diffparamy':
                diff_param_y = float(argv[i + 1])
            elif argv[i] == '-dssppath':
                dssp_path = argv[i + 1]



    if pdb_file_path=='':
        print('Error: The argument -pdb is necessary')
        argument_error = True
    if chain_id=='':
        print('Error: The argument -chainid is necessary')
        argument_error = True
    if dssp_path=='':
        print('Error: The argument -dssppath is necessary')
        argument_error = True

    if (diff_param_x == None and diff_param_y != None) or (diff_param_x != None and diff_param_y == None):
        print('Error: The arguments -diffparamx and -diffparamy should be passed simultaneously')
        argument_error = True

    if argument_error:
        return 'argument error'

    path_error = False

    if not os.path.isfile(pdb_file_path):
        print('Error: PDB file not found')
        path_error = True

    if not os.path.isfile(dssp_path):
        print('Error: DSSP binary not found')
        path_error = True

    if path_error:
        return 'path error'

    pdb_id = ntpath.basename(pdb_file_path)[:4].upper()

    parsed_pdb = parse_pdb(pdb_file_path, pdb_id, chain_id)

    if parsed_pdb == 'invalid chain':
        print('Error: Chain ID is not valid')
        return 'invalid chain'

    aminoacids, aminoacid_ca_coords, aminoacid_letters, aminoacid_resnums = parsed_pdb

    if len(aminoacids) == 0:
        print(pdb_id+'\t'+chain_id+'\t1')
        return pdb_id+'\t'+chain_id+'\t1'
        # quit(0)

    n = len(aminoacids)

    p = sp.Popen([dssp_path, pdb_file_path], stdout=sp.PIPE, stderr=sp.STDOUT)

    dssp = parse_dssp(p.stdout.readlines(), chain_id)
    # dssp2 = DSSP(model, pdb_path, dssp=r'd:/dssp-2.0.4-win32.exe')
    retval = p.wait()

    hbonds_nho, hbonds_ohn = extract_hydrpgen_bonds(dssp, aminoacid_resnums, -0.6)
    hydrogen_bonds = hbonds_nho + hbonds_ohn

    for i in range(n - 1):
        for j in range(i + 1, n):
            if (hydrogen_bonds[i,j] != hydrogen_bonds[j,i]):
                hydrogen_bonds[i,j] = hydrogen_bonds[j,i] = max(hydrogen_bonds[i,j], hydrogen_bonds[j,i])


    beta_bridges = get_beta_bridges(dssp,aminoacid_resnums)

    # Maximum accessible surface area by Miller et al. 1987



    acc = np.zeros(n)

    dssp_resnums = []

    for key in dssp:
        index = aminoacid_resnums.index(dssp[key]['resnum'])
        dssp_resnums.append(dssp[key]['resnum'])
        rel_acc = dssp[key]['acc']/max_asa[dssp[key]['aa']]
        if rel_acc > 1:
            rel_acc = 1
        acc[index] = rel_acc


    if len(dssp) < n:
        # print('DSSP length is lower than PDB!')
        for resnum in aminoacid_resnums:
            if resnum not in dssp_resnums:
                index = aminoacid_resnums.index(resnum)
                acc[index] = 0.5
        # for key in dssp:
        #     if dssp[key]['resnum'] not in aminoacid_resnums:
        #         aminoacid_resnums.index(dssp[key]['resnum'])
        #quit(1)


    sec_struc_labels, betasheet_labels, beta_bridge_indices1, beta_bridge_indices2 = get_sec_struc_info(dssp, aminoacid_resnums)

    alpha_helices = get_alpha_helices(sec_struct_labels=sec_struc_labels)
    beta_strands = get_beta_strands(sec_struc_labels, betasheet_labels)


    co_alpha_helix_matrix = np.zeros([n,n])
    for alpha_helix in alpha_helices:
        for i in range(alpha_helix[0], alpha_helix[1]):
            for j in range(i + 1, alpha_helix[1] + 1):
                co_alpha_helix_matrix[i,j] = co_alpha_helix_matrix[j,i] = 1

    co_beta_strand_matrix = np.zeros([n, n])
    for beta_strand in beta_strands:
        for i in range(beta_strand[0], beta_strand[1]):
            for j in range(i + 1, beta_strand[1] + 1):
                co_beta_strand_matrix[i, j] = co_beta_strand_matrix[j,i] = 1


    hydphob = [hydrophobicity[aminoacid_letters[i]] for i in range(n)]

    graph = make_graph(aminoacids, aminoacid_ca_coords,co_alpha_helix_matrix, co_beta_strand_matrix,betasheet_labels, acc, hydphob,hydrogen_bonds,beta_bridges)

    main_graph = graph.copy()

    contract_alpha_helices(graph, alpha_helices, max_alpha_helix_size_to_contract)

    residue_vartex_map = [None] * n

    for i in range(len(graph.vs)):
        for x in graph.vs[i]['name']:
            residue_vartex_map[x] = i


    num_vtx = len(graph.vs)

    if diff_param_x==None and diff_param_y==None:
        if kernel == 'lap-exp-diff':
            diff_param_x = 0.0105
        elif kernel == 'markov-diff':
            diff_param_x = 0.1024
        elif kernel == 'reg-lap-diff':
            diff_param_x = 0.05
        elif kernel == 'markov-exp-diff':
            diff_param_x = 5
        diff_param_y = 1


    diff_param = diff_param_x * (num_vtx ** diff_param_y)


    if kernel =='lap-exp-diff':
        diff_kernel = lap_exp_diff_kernel(graph, diff_param)
    elif kernel == 'markov-diff':
        diff_kernel = markov_diff_kernel(graph, diff_param)
    elif kernel=='reg-lap-diff':
        diff_kernel = reg_lap_kernel(graph, diff_param)
    elif kernel == 'markov-exp-diff':
        diff_kernel = markov_exp_diff_kernel(graph, diff_param)


    expanded_kernel = np.zeros((n,n))

    for i in range(num_vtx):
        members = graph.vs[i]['name']
        for j in range(len(members)):
            for k in range(j,len(members)):
                p = members[j]
                q = members[k]
                expanded_kernel[p, q] = expanded_kernel[q, p] = diff_kernel[i, i]

        # for p in graph.vs[i]['name']:
        #     for q in graph.vs[i]['name']:
        #         expanded_kernel[p,q] = expanded_kernel[q,p] = diff_kernel[i,i]
        #         print(p,q)

    for i in range(num_vtx - 1):
        for j in range(i+1, num_vtx):
            for p in graph.vs[i]['name']:
                for q in graph.vs[j]['name']:
                    expanded_kernel[p, q] = expanded_kernel[q,p] = diff_kernel[i,j]


    distance_matrix = convert_kernel_to_distance(expanded_kernel, method='norm')

    f = open(os.path.join(here, 'single-multi-classifier.sav'), 'rb')
    weak_classifiers = []
    for i in range(100):
        weak_classifiers.append(pickle.load(f))
    f.close()

    if num_domains == None:

        num_domains = 1
        single_domain_labeling = [0] * n



        degrees = main_graph.strength(weights='weight')

        mean_degree = np.mean(degrees)
        degree_var = np.var(degrees)
        closeness = main_graph.closeness(weights='weight')
        betweenness = main_graph.betweenness(weights='weight')

        degrees_hydphob_corr = np.corrcoef(degrees, hydphob)[0, 1]
        degree_acc_corr = np.corrcoef(degrees, acc)[0, 1]

        closeness_hydphob_corr = np.corrcoef(closeness, hydphob)[0, 1]
        closeness_acc_corr = np.corrcoef(closeness, acc)[0, 1]

        betweenness_hydphob_corr = np.corrcoef(betweenness, hydphob)[0, 1]
        betweenness_acc_corr = np.corrcoef(betweenness, acc)[0, 1]

        if math.isnan(degrees_hydphob_corr):
            degrees_hydphob_corr = 0
        if math.isnan(degree_acc_corr):
            degree_acc_corr = 0
        if math.isnan(closeness_hydphob_corr):
            closeness_hydphob_corr = 0
        if math.isnan(closeness_acc_corr):
            closeness_acc_corr = 0
        if math.isnan(betweenness_hydphob_corr):
            betweenness_hydphob_corr = 0
        if math.isnan(betweenness_acc_corr):
            betweenness_acc_corr = 0

        hydphob_acc_ratio = [(hydphob[i] + max(hydphob)) / ((acc[i] + 0.00001) * 2 * max(hydphob)) for i in
                             range(len(hydphob))]

        core_res_by_hydph_acc_idx = [idx for idx, val in enumerate(hydphob_acc_ratio) if val > 2]
        core_res_hyph_acc_ca_coords = [aminoacid_ca_coords[x] for x in core_res_by_hydph_acc_idx]

        pca = PCA()

        core_hydph_acc_expl_var = [0, 0, 0]

        if len(core_res_by_hydph_acc_idx) >= 3:
            pca.fit(core_res_hyph_acc_ca_coords)
            core_hydph_acc_expl_var = pca.explained_variance_

        num_cores = 0

        if len(core_res_hyph_acc_ca_coords) > 0:

            clusterer = DBSCAN(eps=8, min_samples=6)
            clustering = clusterer.fit(core_res_hyph_acc_ca_coords)

            clustering_labels = set(clustering.labels_)

            if -1 in clustering_labels:
                clustering_labels.remove(-1)

            num_cores = len(clustering_labels)


        feature_vec = np.array([len(main_graph.es) / n, n, mean_degree, degree_var, degrees_hydphob_corr,
                 degree_acc_corr, closeness_hydphob_corr, closeness_acc_corr, betweenness_hydphob_corr,
                 betweenness_acc_corr,num_cores,core_hydph_acc_expl_var[0],core_hydph_acc_expl_var[1],
                 core_hydph_acc_expl_var[2]]).reshape(1, -1)

        predictions = []
        for clf in weak_classifiers:
            predict = clf.predict_proba(feature_vec)
            # predict_train = clf.predict_proba(X)
            predictions.append([p[1] for p in predict])

        predictions = np.array(predictions)
        single_domain_probability = np.mean(predictions, 0)

        labelings = [single_domain_labeling]

        all_assignments = []

        all_assignments.append((1, conv_to_text(
            get_domains_as_segments_by_resnum(get_domains_as_segments(single_domain_labeling), aminoacid_resnums), delimiter=''),
                                '----'))

        opt_num_domains = 1

        if single_domain_probability <= 0.5 or display_all_partiotionings:

            max_sil_score = -1

            while True:
                num_domains += 1
                # result = cluster(num_domains,main_graph, graph,aminoacid_resnums,diff_kernel,min_seg_size,aminoacid_ca_coords, expanded_kernel, seg_numdomians_ratio, sec_struc_labels, residue_vartex_map,aminoacid_letters,acc, beta_strands, max_non_splitted_strand_size, distance_matrix,min_domain_size)
                result = cluster(num_domains, main_graph, graph, aminoacid_resnums, diff_kernel, min_seg_size,
                                 seg_numdomians_ratio, distance_matrix, min_domain_size)

                if result == 'error':
                    num_domains -= 1
                    break
                else:
                    labels, labels_by_vertices, sil_score = result
                    all_assignments.append((num_domains,conv_to_text(get_domains_as_segments_by_resnum(get_domains_as_segments(labels),aminoacid_resnums),delimiter=''), proper_round(sil_score,5)))
                    labelings.append(labels)
                    if sil_score > max_sil_score:
                        max_sil_score = sil_score
                        opt_num_domains = num_domains


        all_assignments[1:] = sorted(all_assignments[1:], key=itemgetter(2), reverse=True)
        # table = Texttable()
        table = PrettyTable(['Num. domains', 'Domain decomposition', 'Sil. score'])
        table.title = 'PDB ID: #PDB_ID#   Chain ID: #CHAIN_ID#'.replace('#PDB_ID#',pdb_id.upper()).replace('#CHAIN_ID#',chain_id.upper())
        if single_domain_probability > 0.5:
            table.add_row(all_assignments[0])

        for assignment in all_assignments[1:]:
            # table.add_row(assignment)
            table.add_row(assignment)

        # table.add_row(all_assignments[0])
        if single_domain_probability <= 0.5:
            table.add_row(all_assignments[0])


        table.align['Domain decomposition']= 'l'
        table.hrules = ALL


        print_format = pdb_id + '\t' + chain_id + '\t' +str(opt_num_domains) + '\t' + conv_to_text(get_domains_as_segments_by_resnum(get_domains_as_segments(labelings[opt_num_domains - 1]), aminoacid_resnums))


        if display_all_partiotionings:

            print(table)

        else:
            print(print_format)

        return print_format

    else:
        result = cluster(num_domains, main_graph, graph, aminoacid_resnums, diff_kernel, min_seg_size,
                                seg_numdomians_ratio, distance_matrix, min_domain_size)
        if result == 'error':
            print('Error: It is impossible to decompose the protein chain by the given parameter values')
            return 'impossible'
        else:
            labels, labels_by_vertices, sil_score = result

            print_format = pdb_id + '\t' + chain_id + '\t' +str(num_domains) + '\t' + conv_to_text(
                    get_domains_as_segments_by_resnum(get_domains_as_segments(labels), aminoacid_resnums))

            print(print_format)
            return print_format



if __name__ == "__main__":
    run(sys.argv[1:])

