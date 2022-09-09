import collections, itertools, math, ntpath, os, pickle, subprocess as sp, numpy as np, sys, warnings
from collections import Counter
from collections import defaultdict
from operator import itemgetter
from igraph import Graph
from prettytable import PrettyTable, ALL
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kernels import *
from protein_structure import *
from tslearn.clustering import KernelKMeans
import re
from sklearn.preprocessing import scale
import unidip.dip as dip
from pyclustertend import hopkins
from sklearn.decomposition import KernelPCA
import bz2
import _pickle as cPickle

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

min_hydrophobicity = min(hydrophobicity.values())
max_hydrophobicity = max(hydrophobicity.values())

rel_hydrophobicity = {aa: (hydrophobicity[aa] - min_hydrophobicity) / (max_hydrophobicity - min_hydrophobicity) for aa in hydrophobicity}

f = bz2.BZ2File(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sm_classifier.pkl.bz2'), 'rb')
clf = cPickle.load(f)
f.close()

def _gcm_(cdf, idxs):
    work_cdf = cdf
    work_idxs = idxs
    gcm = [work_cdf[0]]
    touchpoints = [0]
    while len(work_cdf) > 1:
        distances = work_idxs[1:] - work_idxs[0]
        slopes = (work_cdf[1:] - work_cdf[0]) / distances
        minslope = slopes.min()
        minslope_idx = np.where(slopes == minslope)[0][0] + 1
        gcm.extend(work_cdf[0] + distances[:minslope_idx] * minslope)
        touchpoints.append(touchpoints[-1] + minslope_idx)
        work_cdf = work_cdf[minslope_idx:]
        work_idxs = work_idxs[minslope_idx:]
    return np.array(np.array(gcm)), np.array(touchpoints)


def get_segments(assignment, query=None):
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


def get_shortest_segment_index(segments, segment_labels):
    segment_count = Counter(segment_labels)
    multi_segment_domains = {key: value for key, value in segment_count.items() if value > 1}
    segment_ids_filtered = [x for x in range(len(segments)) if segment_labels[x] in multi_segment_domains]
    # segment_ids_filtered = [int(x) for x in segment_count_filtered.keys()]

    # if len(segment_ids_filtered) > 0:

    # shortest_segment_index = segment_ids_filtered[0]
    shortest_segment_index = 0
    min_length = segments[shortest_segment_index][1] - segments[shortest_segment_index][0] + 1

    # for segment_id in segment_ids_filtered[1:]:
    for segment_id in range(1, len(segments)):
        length = segments[segment_id][1] - segments[segment_id][0] + 1

        if length < min_length:
            shortest_segment_index = segment_id
            min_length = length

    return shortest_segment_index


def get_domains_as_segments(assignment):
    start = 0
    segments = defaultdict(list)

    for i in range(1, len(assignment)):
        if assignment[i] != assignment[i - 1]:
            end = i - 1
            segments[assignment[i - 1]].append((start, end))
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


def conv_to_text(domains, delimiter='\t'):
    # standard_format = str(len(domains))
    standard_format = ''
    for i in domains:
        # if i > 1:
        #     standard_format += '\t'
        if standard_format == '':
            standard_format = '('
        else:
            standard_format += delimiter + '('
        # standard_format += pdbid + chainid + str(i) + ': '
        for j in range(0, len(domains[i])):
            standard_format += str(domains[i][j][0]) + '-' + str(domains[i][j][1])
            if j < len(domains[i]) - 1:
                standard_format += ','
        standard_format += ')'

    return standard_format


def remove_short_segments(assignment, cutoff, distance_matrix=None):
    segments, segment_labels = get_segments(assignment)
    shortest_segment_index = get_shortest_segment_index(segments, segment_labels)
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

                current_segment_res_ids = range(segments[shortest_segment_index][0],
                                                segments[shortest_segment_index][1] + 1)
                pred_segment_res_ids = range(segments[shortest_segment_index - 1][0],
                                             segments[shortest_segment_index - 1][1] + 1)
                succ_segment_res_ids = range(segments[shortest_segment_index + 1][0],
                                             segments[shortest_segment_index + 1][1] + 1)

                # dist_with_pred = average_linkage_distance(current_ca_coords, pred_ca_coords)
                # dist_with_succ = average_linkage_distance(current_ca_coords, succ_ca_coords)

                dist_with_pred = 0
                dist_with_succ = 0
                for i in current_segment_res_ids:
                    for j in pred_segment_res_ids:
                        dist_with_pred += distance_matrix[i, j]
                    for j in succ_segment_res_ids:
                        dist_with_succ += distance_matrix[i, j]

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

        segments, segment_labels = get_segments(assignment)
        shortest_segment_index = get_shortest_segment_index(segments, segment_labels)
        if shortest_segment_index != None:
            shortest_segment = segments[shortest_segment_index]
            shortest_segment_length = shortest_segment[1] - shortest_segment[0] + 1
        else:
            shortest_segment_length = cutoff + 1000


def remove_redundant_segments(labels, num_domains, max_segdom_ratio, distance_matrix):
    segments, segment_labels = get_segments(labels)
    shortest_segment_index = get_shortest_segment_index(segments, segment_labels)
    if shortest_segment_index != None:
        shortest_segment = segments[shortest_segment_index]
    else:
        return None

    while len(segments) / float(num_domains) > max_segdom_ratio:
        remove_short_segments(labels, shortest_segment[1] - shortest_segment[0] + 2, distance_matrix)
        segments, segment_labels = get_segments(labels)
        shortest_segment_index = get_shortest_segment_index(segments, segment_labels)
        if shortest_segment_index is not None:
            shortest_segment = segments[shortest_segment_index]
        else:
            break


def get_small_segments_idx(segments, min_seg_size):
    segments_size = [segments[i][1] - segments[i][0] + 1 for i in range(len(segments))]

    small_segments_idx = []
    for i in range(len(segments_size)):
        if segments_size[i] < min_seg_size:
            small_segments_idx.append(i)

    return small_segments_idx


def w_sil_score(dist_matrix, cluster_labels, weights):

    a = np.ones(len(cluster_labels)) * float('inf')
    b = np.ones(len(cluster_labels)) * float('inf')

    for label in set(cluster_labels):
        indexes_in = np.where(cluster_labels == label)[0]
        d = dist_matrix[indexes_in,:][:,indexes_in]
        w = weights[indexes_in]
        for i in range(d.shape[0]):
            d_i = np.delete(d[i,:], i)
            w_i = np.delete(w, i)
            a_i = np.average(d_i, weights= w_i)
            a[indexes_in[i]] = a_i

        indexes_out = np.where(cluster_labels != label)[0]
        d_out = dist_matrix[indexes_out, :][:, indexes_in]
        for i in range(d_out.shape[0]):
            d_i = d_out[i,:]
            b_i = np.average(d_i, weights=w)
            b[indexes_out[i]] = min(b[indexes_out[i]], b_i)

    sil = (b - a) /  np.vstack([a, b]).max(axis=0)

    return sil.mean()

def cluster(num_domains, diff_kernel, min_seg_size, max_segdom_ratio, distance_matrix, min_domain_size,
            clustering_method, alpha_helices, max_alpha_helix_size_to_merge, hydphob):
    # try:

    if clustering_method == 'SP':
        no_err = False
        random_state = 0
        while not no_err:
            try:
                clustering = SpectralClustering(n_clusters=num_domains, assign_labels="kmeans", random_state=random_state,
                                                affinity='precomputed', n_init=100).fit(diff_kernel)
                no_err = True
                # print('ok')
            except:
                # print('error')
                random_state += 1
                # pass

    elif clustering_method == 'KK':
        no_err = False
        random_state = 0
        while not no_err:
            try:
                clustering = KernelKMeans(n_clusters=num_domains, random_state=random_state, n_init=100, kernel='precomputed').fit(
                    diff_kernel)

                no_err = True
            except:
                random_state += 1
                # pass


    labels = clustering.labels_.copy()

    for alpha_helix in alpha_helices:
        if alpha_helix[1] - alpha_helix[0] + 1 <= max_alpha_helix_size_to_merge:
            alpha_helix_labels = labels[alpha_helix[0]:alpha_helix[1] + 1]
            counter = collections.Counter(alpha_helix_labels)
            if len(counter) > 1:
                most_common = counter.most_common(1)[0][0]
                labels[alpha_helix[0]:alpha_helix[1] + 1] = [most_common] * (alpha_helix[1] - alpha_helix[0] + 1)

    remove_short_segments(labels, min_seg_size, distance_matrix)

    remove_redundant_segments(labels, num_domains, max_segdom_ratio, distance_matrix)

    hydphob_res_mask = hydphob > 2

    if (len(set(labels[hydphob_res_mask])) < num_domains):
        return 'error'


    sil_score = silhouette_score(distance_matrix[hydphob_res_mask,:][:,hydphob_res_mask], labels=labels[hydphob_res_mask], metric="precomputed")


    for label in set(labels):
        if np.count_nonzero(labels == label) < min_domain_size:
            return 'error'

    return labels, labels, sil_score

    # except:
    #     return 'error'


def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.') + dec + 2]
    if num[-1] >= '5':
        a = num[:-2 - (not dec)]  # integer part
        b = int(num[-2 - (not dec)]) + 1  # decimal part
        return float(a) + b ** (-dec + 1) if a and b == 10 else float(a + str(b))
    return float(num[:-1])


def multi_domain_assignment(min_num_doms, max_num_doms, aminoacid_resnums, hydphob, kernel_matrix, min_seg_size, max_segdom_ratio, distance_matrix,
                         min_domain_size, clustering_method, alpha_helices, max_alpha_helix_size_to_contract):

    multidomain_assignments = []

    multidomain_labelings = []

    opt_num_domains = m = min_num_doms

    max_sil_score = -1

    while m <= max_num_doms:

        result = cluster(m, kernel_matrix, min_seg_size, max_segdom_ratio, distance_matrix,
                         min_domain_size, clustering_method, alpha_helices, max_alpha_helix_size_to_contract, hydphob)

        if result == 'error':
            break
        else:
            labels, labels_by_vertices, sil_score = result
            multidomain_assignments.append((m, conv_to_text(
                get_domains_as_segments_by_resnum(get_domains_as_segments(labels), aminoacid_resnums),
                delimiter=''), proper_round(sil_score, 5)))
            multidomain_labelings.append(labels)
            if sil_score > max_sil_score:
                max_sil_score = sil_score
                opt_num_domains = m
            m += 1

    if len(multidomain_assignments)== 0:
        opt_num_domains = 0

    return multidomain_assignments, multidomain_labelings, opt_num_domains

def make_table(pdb_id, chain_id, multidomain_assignments_sorted, singledomain_assignment = None, singledomain_assignment_pos = 'top'):
    table = PrettyTable(['Num. domains', 'Domain decomposition', 'Sil. score'])
    table.title = f'PDB ID: {pdb_id.upper()}   Chain ID: {chain_id.upper()}'
    if singledomain_assignment is not None and singledomain_assignment_pos=='top':
        table.add_row(singledomain_assignment)
    for assignment in multidomain_assignments_sorted:
        table.add_row(assignment)
    if singledomain_assignment is not None and singledomain_assignment_pos == 'bottom':
        table.add_row(singledomain_assignment)
    table.align['Domain decomposition'] = 'l'
    table.hrules = ALL
    return table

help_text = """

  KluDo (Diffusion Kernel-based Graph Node Clustering for Protein Domain
  Assignment), is an automatic framework that incorporates diffusion kernels
  on protein graphs as affinity measures between residues to decompose protein
  structures into structural domains.
  
  Here is the list of arguments:

  --help                     Help
  --pdb [PATH]               PDB file Path (*)
  --chainid [ID]             Chain ID (*)
  --dssppath [PATH]          DSSP binary file path
  --clustering [METHOD]      The clustering method
  --numdomains [NUMBER]      The number of domains
  --minsegsize [SIZE]        Minimum segment size
  --mindomainsize [SIZE]     Minimum domain size
  --maxalphahelix [SIZE]     Maximum size of alpha-helix to contract
  --maxsegdomratio [RATIO]   Maximum ratio of segment count to domain count
  --kernel [TYPE]            The type of graph node kernel (**)
  --dispall                  Display all candidate partitionings
  --bw_a [VALUE]             Bandwidth parameter x (***)
  --bw_b [VALUE]             Bandwidth parameter y (***)
 
  *
  These arguments are necessary
 
  **
  Type should be choosen from:
   LED
   MD
   MED
   RL
   

  ***
  The parameters bw_a and bw_b are coefficient (x) and exponent (y)
  of the protein size (n) respectively, which determine the bandwidth
  parameter (β or t) of each kernel. (xn^y)
"""


def predict(clf, X_test, thr):
    y_pred = np.array(['M'] * X_test.shape[0])
    y_pred_proba = clf.predict_proba(X_test)
    s = y_pred_proba[:,np.where(clf.classes_== 'S')[0][0]]
    y_pred[s > thr] = 'S'

    return y_pred

def run(pdb_file_path, chain_id, num_domains=(1,99), min_seg_size=27, max_alpha_helix_size_to_contract=30,
        max_segdom_ratio=1.5, min_domain_size=27, kernel='LED', dispall=False,
        bw_a=None, bw_b=None, dssp_path='/usr/bin/dssp', clustering_method='SP'):
    output = ''

    err = False

    if not os.path.isfile(pdb_file_path):
        output += f'Error: PDB file not found ({pdb_file_path})\n'
        err = True
    else:
        pdb_id = ntpath.basename(pdb_file_path)[:4].upper()
        parsed_pdb = parse_pdb(pdb_file_path, chain_id)
        if parsed_pdb == 'invalid chain':
            output += f'Error: Chain ID is not valid ({chain_id})\n'
            err = True

    if not os.path.isfile(dssp_path):
        output += f'Error: DSSP binary not found ({dssp_path})'
        err = True

    if clustering_method not in {'SP', 'KK'}:
        output += f'Error: Invalid clustering method ({clustering_method})\n'
        err = True

    if kernel not in {'LED', 'MD', 'RL', 'MED'}:
        output += f'Error: Invalid kernel: {kernel}\n'
        err = True

    if (bw_a == None and bw_b != None) or (bw_a != None and bw_b == None):
        output += 'Error: The arguments --bw_a and --bw_b should be passed simultaneously\n'
        err = True

    if type(num_domains) is tuple:
        if len(num_domains) == 2:
            if num_domains[0]>=num_domains[1]:
                output += 'Error: wrong range for --numdomains\n'
                err = True
            if num_domains[0]>99 or num_domains[1]>99:
                output += 'Error: --numdomains out of possible range\n'
            if num_domains[0] < 1 or num_domains[1] < 1:
                output += 'Error: --numdomains out of possible range\n'
        else:
            output += 'Error: wrong range for --numdomains\n'
            err = True
    elif type(num_domains) is int:
        if num_domains>99 or num_domains <1:
            output += 'Error: wrong value for --numdomains\n'
            err = True
    else:
        output += 'Error: wrong input type for --numdomains\n'
        err = True

    if err:
        return output

    aminoacids, atoms, aminoacid_ca_coords, aminoacid_letters, aminoacid_resnums, radius_of_gyration = parsed_pdb

    if len(aminoacids) == 0:
        # print(pdb_id + '\t' + chain_id + '\t1')
        return pdb_id + '\t' + chain_id + '\t1'
        # quit(0)

    n = len(aminoacids)

    p = sp.Popen([dssp_path, pdb_file_path], stdout=sp.PIPE, stderr=sp.STDOUT)

    dssp = parse_dssp(p.stdout.readlines(), chain_id)
    # dssp2 = DSSP(model, pdb_path, dssp=r'd:/dssp-2.0.4-win32.exe')
    retval = p.wait()

    # hbonds_nho, hbonds_ohn = extract_hydrpgen_bonds(dssp, aminoacid_resnums, -0.6)
    # hydrogen_bonds = hbonds_nho + hbonds_ohn
    #
    # for i in range(n - 1):
    #     for j in range(i + 1, n):
    #         if (hydrogen_bonds[i, j] != hydrogen_bonds[j, i]):
    #             hydrogen_bonds[i, j] = hydrogen_bonds[j, i] = max(hydrogen_bonds[i, j], hydrogen_bonds[j, i])
    #
    # beta_bridges = get_beta_bridges(dssp, aminoacid_resnums)

    # Maximum accessible surface area by Miller et al. 1987



    acc = np.zeros(n)

    dssp_resnums = []

    for key in dssp:
        index = aminoacid_resnums.index(dssp[key]['resnum'])
        dssp_resnums.append(dssp[key]['resnum'])
        rel_acc = dssp[key]['acc'] / max_asa[dssp[key]['aa']]
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
                # quit(1)

    sec_struc_labels, betasheet_labels, beta_bridge_indices1, beta_bridge_indices2 = get_sec_struc_info(dssp,
                                                                                                        aminoacid_resnums)

    alpha_helices = get_alpha_helices(sec_struct_labels=sec_struc_labels)
    beta_strands = get_beta_strands(sec_struc_labels, betasheet_labels)

    co_alpha_helix_matrix = np.zeros([n, n])
    for alpha_helix in alpha_helices:
        for i in range(alpha_helix[0], alpha_helix[1]):
            for j in range(i + 1, alpha_helix[1] + 1):
                co_alpha_helix_matrix[i, j] = co_alpha_helix_matrix[j, i] = 1

    co_beta_strand_matrix = np.zeros([n, n])
    for beta_strand in beta_strands:
        for i in range(beta_strand[0], beta_strand[1]):
            for j in range(i + 1, beta_strand[1] + 1):
                co_beta_strand_matrix[i, j] = co_beta_strand_matrix[j, i] = 1

    hydphob = np.array([hydrophobicity[aminoacid_letters[i]] for i in range(n)])
    rel_hydphob = np.array([rel_hydrophobicity[aminoacid_letters[i]] for i in  range(n)])

    # graph = make_graph_serial(aminoacids, aminoacid_ca_coords, co_alpha_helix_matrix, co_beta_strand_matrix, betasheet_labels,
    #                    acc, hydphob, hydrogen_bonds, beta_bridges)

    graph = make_graph(atoms)

    # num_vtx = len(graph.vs)

    if bw_a == None and bw_b == None:
        if kernel == 'LED':
            if clustering_method == 'SP':
                bw_a = 0.004
            else:
                bw_a = 0.006
        elif kernel == 'MD':
            if clustering_method == 'SP':
                bw_a = 0.8
            else:
                bw_a = 0.25
        elif kernel == 'RL':
            if clustering_method == 'SP':
                bw_a = 0.022
            else:
                bw_a = 0.021
        elif kernel == 'MED':
            if clustering_method == 'SP':
                bw_a = 0.35
            else:
                bw_a = 0.4

        bw_b = 2




    if type(num_domains) is tuple:

        if num_domains[0] == 1:

            singledomain_labeling = [0] * n
            singledomain_result = conv_to_text(get_domains_as_segments_by_resnum(get_domains_as_segments(singledomain_labeling), aminoacid_resnums))
            singledomain_assignment = (1, singledomain_result , '----')


            feature_vec = extract_features(aminoacids, aminoacid_ca_coords, radius_of_gyration, n,graph, hydphob,rel_hydphob, acc)

            if len(feature_vec) != 78:
                return 'Error: issue in feature extraction'



            # f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sm_classifier_zandieh.pkl'), 'rb')
            # clf = pickle.load(f)
            # f.close()

            predicted_as_singledomain = clf.predict([feature_vec])[0] == 'S'
            # predicted_as_singledomain = (clf.predict_proba([feature_vec])[0][1]>0.55).astype(bool)

            # dm_hydph = distance_matrix[hydphob>2,:][:,hydphob>2]

            # dm_hydph_triu = dm_hydph[np.triu_indices(dm_hydph.shape[0], k=1)]
            #

            # diptst_stat, diptst_pval, _ = dip.diptst(np.histogram(pc.ravel()[hydphob > 2], density=True, bins=10)[0], is_hist=True)
            # diptst_stat, diptst_pval, _ = dip.diptst(np.histogram(distance_matrix_triu, density=True, bins=10)[0], is_hist=True)

            # predicted_as_singledomain = diptst_pval > 0.5



            # predicted_as_singledomain = predict(clf,np.array([feature_vec]),0.7)[0] == 'S'

            if predicted_as_singledomain:



                if dispall:
                    # kernel_matrix = calc_kernel(graph, bw, kernel)

                    # bw = bw_a * (num_vtx ** bw_b)
                    bw = bw_a * (radius_of_gyration ** bw_b)

                    kernel_matrix = calc_kernel(graph, kernel, bw)

                    if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
                        output += "Error: Too large bandwidth parameter\n"
                        err = True

                    if err:
                        return output

                    distance_matrix = convert_kernel_to_distance(kernel_matrix, method='norm')

                    params = dict()
                    params['kernel_matrix'] = kernel_matrix
                    params['min_seg_size'] = min_seg_size
                    params['max_segdom_ratio'] = max_segdom_ratio
                    params['distance_matrix'] = distance_matrix
                    params['min_domain_size'] = min_domain_size
                    params['clustering_method'] = clustering_method
                    params['alpha_helices'] = alpha_helices
                    params['max_alpha_helix_size_to_contract'] = max_alpha_helix_size_to_contract

                    multidomain_assignments, multidomain_labelings, opt_num_domains = multi_domain_assignment(num_domains[0] + 1, num_domains[1], aminoacid_resnums, hydphob, **params)
                    multidomain_assignments_sorted = sorted(multidomain_assignments, key=itemgetter(2), reverse=True)
                    # multidomain_assignments.insert()
                    output = make_table(pdb_id, chain_id, multidomain_assignments_sorted, singledomain_assignment, 'top')
                    return output

                else:
                    output =f'{pdb_id}\t{chain_id}\t1\t{singledomain_result}'
                    return output
            else:
                try:
                    # kernel_matrix = calc_kernel(graph, bw, kernel)

                    # bw = bw_a * (num_vtx ** bw_b)
                    bw = bw_a * (radius_of_gyration ** bw_b)

                    kernel_matrix = calc_kernel(graph, kernel, bw)

                    if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
                        output += "Error: Too large bandwidth parameter\n"
                        err = True

                    if err:
                        return output

                    distance_matrix = convert_kernel_to_distance(kernel_matrix, method='norm')

                    params = dict()
                    params['kernel_matrix'] = kernel_matrix
                    params['min_seg_size'] = min_seg_size
                    params['max_segdom_ratio'] = max_segdom_ratio
                    params['distance_matrix'] = distance_matrix
                    params['min_domain_size'] = min_domain_size
                    params['clustering_method'] = clustering_method
                    params['alpha_helices'] = alpha_helices
                    params['max_alpha_helix_size_to_contract'] = max_alpha_helix_size_to_contract
                    multidomain_assignments, multidomain_labelings, opt_num_domains = multi_domain_assignment(num_domains[0] + 1, num_domains[1], aminoacid_resnums, hydphob, **params)
                    multidomain_assignments_sorted = sorted(multidomain_assignments, key=itemgetter(2), reverse=True)

                    if dispall:
                        output = make_table(pdb_id, chain_id, multidomain_assignments_sorted, singledomain_assignment, 'bottom')
                        return output
                    else:
                        if len(multidomain_assignments) > 0:
                            multidomain_result = conv_to_text(get_domains_as_segments_by_resnum(get_domains_as_segments(multidomain_labelings[opt_num_domains - 2]), aminoacid_resnums))
                            output = f'{pdb_id}\t{chain_id}\t{opt_num_domains}\t{multidomain_result}'
                            return output
                        else: # reject
                            output = f'{pdb_id}\t{chain_id}\t1\t{singledomain_result}'
                            return output
                except:
                    print('error', pdb_id, chain_id)

        else: # numdomains --> [2, infinity]
            # kernel_matrix = calc_kernel(graph, bw, kernel)

            # bw = bw_a * (num_vtx ** bw_b)
            bw = bw_a * (radius_of_gyration ** bw_b)

            kernel_matrix = calc_kernel(graph, kernel, bw)

            if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
                output += "Error: Too large bandwidth parameter\n"
                err = True

            if err:
                return output

            distance_matrix = convert_kernel_to_distance(kernel_matrix, method='norm')

            params = dict()
            params['kernel_matrix'] = kernel_matrix
            params['min_seg_size'] = min_seg_size
            params['max_segdom_ratio'] = max_segdom_ratio
            params['distance_matrix'] = distance_matrix
            params['min_domain_size'] = min_domain_size
            params['clustering_method'] = clustering_method
            params['alpha_helices'] = alpha_helices
            params['max_alpha_helix_size_to_contract'] = max_alpha_helix_size_to_contract

            multidomain_assignments, multidomain_labelings, opt_num_domains = multi_domain_assignment(num_domains[0], num_domains[1], aminoacid_resnums, hydphob, **params)

            if len(multidomain_assignments) == 0:
                output += 'Error: It is impossible to decompose the protein chain by the given parameter values\n'
                err = True
            else:

                if dispall:
                    multidomain_assignments_sorted = sorted(multidomain_assignments, key=itemgetter(2), reverse=True)
                    output = make_table(pdb_id, chain_id, multidomain_assignments_sorted)
                    return output
                else:
                    multidomain_result = conv_to_text(get_domains_as_segments_by_resnum(get_domains_as_segments(multidomain_labelings[opt_num_domains - 2]), aminoacid_resnums))
                    output = f'{pdb_id}\t{chain_id}\t{multidomain_assignments[opt_num_domains - 2][0]}\t{multidomain_result}'
                    return output


    else:
        if num_domains==1:
            singledomain_labeling = [0] * n
            singledomain_result = conv_to_text(get_domains_as_segments_by_resnum(get_domains_as_segments(singledomain_labeling), aminoacid_resnums))
            output = f'{pdb_id}\t{chain_id}\t1\t{singledomain_result}'

            return output

        else:
            # kernel_matrix = calc_kernel(graph, bw, kernel)

            # bw = bw_a * (num_vtx ** bw_b)
            bw = bw_a * (radius_of_gyration ** bw_b)

            kernel_matrix = calc_kernel(graph, kernel, bw)

            if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
                output += "Error: Too large bandwidth parameter\n"
                err = True

            if err:
                return output

            distance_matrix = convert_kernel_to_distance(kernel_matrix, method='norm')

            params = dict()
            params['kernel_matrix'] = kernel_matrix
            params['min_seg_size'] = min_seg_size
            params['max_segdom_ratio'] = max_segdom_ratio
            params['distance_matrix'] = distance_matrix
            params['min_domain_size'] = min_domain_size
            params['clustering_method'] = clustering_method
            params['alpha_helices'] = alpha_helices
            params['max_alpha_helix_size_to_contract'] = max_alpha_helix_size_to_contract

            result = cluster(num_domains, kernel_matrix, min_seg_size, max_segdom_ratio, distance_matrix,
                             min_domain_size, clustering_method, alpha_helices, max_alpha_helix_size_to_contract, hydphob)
            err = result == 'error'
            if err:
                output += 'Error: It is impossible to decompose the protein chain by the given parameter values\n'
                err = True
            else:
                labels, labels_by_vertices, sil_score = result

                output = pdb_id + '\t' + chain_id + '\t' + str(num_domains) + '\t' + conv_to_text(
                    get_domains_as_segments_by_resnum(get_domains_as_segments(labels), aminoacid_resnums))

                return output

    if err:
        return output


def weighted_variance(points, w):

    if type(w) is not np.ndarray:
        w = np.array(w)

    v1 = w.sum()
    v2 = (w ** 2).sum()

    w_var = np.dot(((points - points.mean(axis=0, keepdims= True)) ** 2).T, w) / (v1 - (v2 / v1))

    return w_var



def extract_features(aminoacids, aminoacid_ca_coords, radius_of_gyration, n, graph, hydphob, rel_hydphob, acc):


    import numpy as np

    np.random.seed(0)


    ca_coords_centered = scale(np.array(aminoacid_ca_coords), with_std=False)

    pca = PCA()

    pca.fit(ca_coords_centered)
    pca_ca_coords_centered = pca.transform(ca_coords_centered)
    pca_expl_var = pca.explained_variance_


    hopkins_scores = [hopkins(ca_coords_centered, len(aminoacids)) for _ in range(100)]

    hopkins_score = np.mean(hopkins_scores)

    bins_pc1 = np.arange(pca_ca_coords_centered[:, 0].min(), pca_ca_coords_centered[:, 0].max() + 4, step=4)
    bins_pc2 = np.arange(pca_ca_coords_centered[:, 1].min(), pca_ca_coords_centered[:, 1].max() + 4, step=4)
    bins_pc3 = np.arange(pca_ca_coords_centered[:, 2].min(), pca_ca_coords_centered[:, 2].max() + 4, step=4)

    hist_pc1 = np.histogram(pca_ca_coords_centered[:, 0], bins=bins_pc1)[0] / n
    hist_pc2 = np.histogram(pca_ca_coords_centered[:, 1], bins=bins_pc2)[0] / n
    hist_pc3 = np.histogram(pca_ca_coords_centered[:, 2], bins=bins_pc3)[0] / n

    dip_stat_pc1, diptest_pval_pc1, diptest_indices_pc1 = dip.diptst(hist_pc1, is_hist=True)
    dip_stat_pc2, diptest_pval_pc2, diptest_indices_pc2 = dip.diptst(hist_pc2, is_hist=True)
    dip_stat_pc3, diptest_pval_pc3, diptest_indices_pc3 = dip.diptst(hist_pc3, is_hist=True)

    clust_coef_g = graph.transitivity_undirected(mode="zero")
    clust_coef_al = graph.transitivity_avglocal_undirected(mode="zero")
    clust_coef_al_w = graph.transitivity_avglocal_undirected(mode="zero",weights="weight")

    degree = graph.degree()
    w_degree = graph.strength(weights='weight')

    degree_hydphob_corr = np.corrcoef(degree, hydphob)[0, 1]
    w_degree_hydphob_corr = np.corrcoef(w_degree, hydphob)[0, 1]

    degree_acc_corr = np.corrcoef(degree, acc)[0, 1]
    w_degree_acc_corr = np.corrcoef(w_degree, acc)[0, 1]

    # if math.isnan(degree_acc_corr):
    #     degree_acc_corr = 0

    closeness = graph.closeness()
    w_closeness = graph.closeness(weights='weight')

    closeness_hydphob_corr = np.corrcoef(closeness, hydphob)[0, 1]
    w_closeness_hydphob_corr = np.corrcoef(w_closeness, hydphob)[0, 1]

    closeness_acc_corr = np.corrcoef(closeness, acc)[0, 1]
    w_closeness_acc_corr = np.corrcoef(w_closeness, acc)[0, 1]

    # if math.isnan(closeness_acc_corr):
    #     closeness_acc_corr = 0

    betweenness = graph.betweenness()
    w_betweenness = graph.betweenness(weights='weight')

    betweenness_hydphob_corr = np.corrcoef(betweenness, hydphob)[0, 1]
    w_betweenness_hydphob_corr = np.corrcoef(w_betweenness, hydphob)[0, 1]

    betweenness_acc_corr = np.corrcoef(betweenness, acc)[0, 1]
    w_betweenness_acc_corr = np.corrcoef(w_betweenness, acc)[0, 1]


    e = len(graph.es)
    w_e = sum(graph.es['weight'])

    e_n_ratio = e / n
    w_e_n_ratio = w_e / n


    pca_w_var_degree = weighted_variance(pca_ca_coords_centered, degree)
    pca_w_var_w_degree = weighted_variance(pca_ca_coords_centered, w_degree)

    pca_w_var_closeness = weighted_variance(pca_ca_coords_centered, closeness)
    pca_w_var_w_closeness = weighted_variance(pca_ca_coords_centered, w_closeness)

    pca_w_var_betweenness = weighted_variance(pca_ca_coords_centered, betweenness)
    pca_w_var_w_betweenness = weighted_variance(pca_ca_coords_centered, w_betweenness)

    pca_w_var_acc = weighted_variance(pca_ca_coords_centered, 1 - acc)
    pca_w_var_hydphob = weighted_variance(pca_ca_coords_centered, rel_hydphob)

    degree_mean, degree_var = np.mean(degree), np.var(degree)
    w_degree_mean, w_degree_var = np.mean(w_degree), np.var(w_degree)
    closeness_mean, closeness_var = np.mean(closeness), np.var(closeness)
    w_closeness_mean, w_closeness_var = np.mean(w_closeness), np.var(w_closeness)
    betweenness_mean, betweenness_var = np.mean(betweenness), np.var(betweenness)
    w_betweenness_mean, w_betweenness_var = np.mean(w_betweenness), np.var(w_betweenness)
    acc_mean, acc_var = np.mean(acc), np.var(acc)
    hydphob_mean, hydphob_var = np.mean(hydphob), np.var(hydphob)


    # graph.laplacian(weights='weight')
    L = graph.laplacian(weights='weight', normalized=True)

    spectrum = np.linalg.eigvals(L)
    spectrum.sort()

    f_vec1 = [n, e, w_e, e_n_ratio, w_e_n_ratio, radius_of_gyration]

    f_vec2 = [dip_stat_pc1, dip_stat_pc2, dip_stat_pc3, hopkins_score, clust_coef_g, clust_coef_al, clust_coef_al_w]

    f_vec3 = [pca_expl_var[0], pca_expl_var[1], pca_expl_var[2], pca_w_var_degree[0], pca_w_var_degree[1],
              pca_w_var_degree[2], pca_w_var_w_degree[0], pca_w_var_w_degree[1], pca_w_var_w_degree[2],
              pca_w_var_closeness[0], pca_w_var_closeness[1], pca_w_var_closeness[2], pca_w_var_w_closeness[0],
              pca_w_var_w_closeness[1], pca_w_var_w_closeness[2], pca_w_var_betweenness[0], pca_w_var_betweenness[1],
              pca_w_var_betweenness[2], pca_w_var_w_betweenness[0], pca_w_var_w_betweenness[1],
              pca_w_var_w_betweenness[2],
              pca_w_var_acc[0], pca_w_var_acc[1], pca_w_var_acc[2], pca_w_var_hydphob[0], pca_w_var_hydphob[1],
              pca_w_var_hydphob[2]]

    f_vec4 = [degree_mean, degree_var, w_degree_mean, w_degree_var, closeness_mean, closeness_var,
              w_closeness_mean, w_closeness_var, betweenness_mean, betweenness_var, w_betweenness_mean,
              w_betweenness_var, acc_mean, acc_var, hydphob_mean, hydphob_var]

    f_vec4 = [0 if math.isnan(x) else x for x in f_vec4]

    f_vec5 = [degree_hydphob_corr, w_degree_hydphob_corr, degree_acc_corr, w_degree_acc_corr,
              closeness_hydphob_corr, w_closeness_hydphob_corr, closeness_acc_corr, w_closeness_acc_corr,
              betweenness_hydphob_corr, w_betweenness_hydphob_corr, betweenness_acc_corr, w_betweenness_acc_corr]

    f_vec5 = [0 if math.isnan(x) else x for x in f_vec5]

    f_vec6 = spectrum[1:11].tolist()


    f_vec = f_vec1 + f_vec2 + f_vec3 + f_vec4 + f_vec5 + f_vec6


    return f_vec


if __name__ == "__main__":

    params = dict()

    unknown_args = []
    nonnumeric_val_args = []

    numdomains_format_err = False
    numdomains_range_err= False

    help = False

    for i in range(1, len(sys.argv)):
        if sys.argv[i][0:2] == '--':
            if sys.argv[i] == '--help':
                help = True
                break
            elif sys.argv[i] == '--pdb':
                params['pdb_file_path'] = sys.argv[i + 1]
            elif sys.argv[i] == '--chainid':
                params['chain_id'] = sys.argv[i + 1].upper()
            elif sys.argv[i] == '--numdomains':
                if bool(re.match("(^(([1-9][0-9]?)|())-(([1-9][0-9]?)|())$)|(^([1-9][0-9]?)$)", sys.argv[i+1])):
                    if sys.argv[i + 1].isdigit():
                        params['num_domains'] = int(sys.argv[i + 1])
                    elif bool(re.match("^([1-9][0-9]?)-([1-9][0-9]?)$", sys.argv[i+1])):
                        rng = [int(x) for x in sys.argv[i+1].split('-')]
                        if rng[0] >= rng[1]:
                            numdomains_range_err = True
                        else:
                            params['num_domains'] = (rng[0], rng[1])
                    elif bool(re.match('^()-([1-9][0-9]?)$', sys.argv[i+1])):
                        x = int(sys.argv[i + 1][1:])
                        if x > 1:
                            rng = (1, x)
                            params['num_domains'] = (rng[0], rng[1])
                        else:
                            numdomains_range_err = True
                    elif bool(re.match("^([1-9][0-9]?)-()$", sys.argv[i+1])):
                        x = int(sys.argv[i + 1][:-1])
                        rng = (x, 99)
                        params['num_domains'] = (rng[0], rng[1])
                    #else: #sys.argv[i+1]== '-':
                else:
                    numdomains_format_err = True
                # try:
                #     params['num_domains'] = sys.argv[i + 1]
                # except:
                #     nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--minsegsize':
                try:
                    params['min_seg_size'] = int(sys.argv[i + 1])
                except:
                    nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--mindomainsize':
                try:
                    params['min_domain_size'] = int(sys.argv[i + 1])
                except:
                    nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--maxalphahelix':
                try:
                    params['max_alpha_helix_size_to_contract'] = int(sys.argv[i + 1])
                except:
                    nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--maxsegdomratio':
                try:
                    params['max_segdom_ratio'] = float(sys.argv[i + 1])
                except:
                    nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--kernel':
                params['kernel'] = sys.argv[i + 1]
            elif sys.argv[i] == '--dispall':
                params['dispall'] = True
            elif sys.argv[i] == '--bw_a':
                try:
                    params['bw_a'] = float(sys.argv[i + 1])
                except:
                    nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--bw_b':
                try:
                    params['bw_b'] = float(sys.argv[i + 1])
                except:
                    nonnumeric_val_args.append(sys.argv[i])
            elif sys.argv[i] == '--dssppath':
                params['dssp_path'] = sys.argv[i + 1]
            elif sys.argv[i] == '--clustering':
                params['clustering_method'] = sys.argv[i + 1]
            # elif sys.argv[i] == '--pgraph':
            #     params['para_graph_const'] = True
            else:
                unknown_args.append(sys.argv[i])

    if help:
        print(help_text)
    else:

        err = False

        if 'pdb_file_path' not in params:
            err = True
            print('Error: The argument --pdb is necessary')

        if 'chain_id' not in params:
            err = True
            print('Error: The argument --chainid is necessary\n')

        if len(unknown_args) > 0:
            err = True
            args = ', '.join(unknown_args)
            print(f'Error: Unknown argument(s): {args}')

        if len(nonnumeric_val_args) > 0:
            err = True
            args = ', '.join(nonnumeric_val_args)
            print(f'Error: Non-numeric value(s) for the numeric argument(s): {args}')

        if numdomains_format_err or numdomains_range_err:
            err = True
            print('Error: wrong input for the argument --numdomains')


        if not err:
            #output = run(**params)
            #print(output)
            try:
                output = run(**params)
                print(output)
            except TypeError as type_err:
                print('Error: ' + type_err.args[0])
