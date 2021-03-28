import collections, itertools, math, ntpath, os, pickle, subprocess as sp, numpy as np, sys, warnings
from collections import Counter
from collections import defaultdict
from operator import itemgetter
from igraph import Graph
from prettytable import PrettyTable, ALL
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from kernels import *
from protein_structure import *
from kernel_kmeans import KernelKMeans
from single_multi_classifier import SingleMultiClassifier

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

    loaded_model = pickle.load(open(os.path.join(here, 'edge_weight_predictor.sav'), 'rb'))

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


                    result = loaded_model.predict_proba(np.array([num_bb_contacts,num_all_contacts ,co_beta_strand_matrix[i,j],co_alpha_helix_matrix[i,j],hydrogen_bonds[i,j],beta_bridges[i,j],in_same_betasheet,beta_bridge_in_same_beta_sheet,mean_relacc,mean_hphob,j-i,ca_dist]).reshape(1, -1))[0]


                    g.add_edge(i, j, weight = result[1])


    return g


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


def cluster(num_domains,graph, aminoacid_resnums, diff_kernel, min_seg_size, seg_numdomians_ratio,distance_matrix,min_domain_size, clustering_method, alpha_helices, max_alpha_helix_size_to_merge):

    try:

        if clustering_method == 'spectral':
            clustering = SpectralClustering(n_clusters=num_domains, assign_labels="kmeans", random_state=0,
                                            affinity='precomputed', n_init=100).fit(diff_kernel)
        else:
            clustering = KernelKMeans(n_clusters=num_domains, random_state=0, kernel='precomputed').fit(
                diff_kernel)


        clusters_by_resnum, clusters_by_vtx_index, clusters_by_index = get_clusters_by_vtx_labels(graph,aminoacid_resnums,clustering.labels_)

        labels = np.zeros(len(aminoacid_resnums),dtype=int)



        # labels = [None] * len(aminoacid_resnums)

        for i in range(len(clusters_by_index)):
            labels[clusters_by_index[i]] = i


        for alpha_helix in alpha_helices:
            if alpha_helix[1] - alpha_helix[0] + 1 <= max_alpha_helix_size_to_merge:
                alpha_helix_labels = labels[alpha_helix[0]:alpha_helix[1] + 1]
                counter = collections.Counter(alpha_helix_labels)
                if len(counter)>1:
                    most_common = counter.most_common(1)[0][0]
                    labels[alpha_helix[0]:alpha_helix[1] + 1] = [most_common] * (alpha_helix[1] - alpha_helix[0] + 1)



        remove_short_segments(labels, min_seg_size, distance_matrix)

        remove_redundant_segments(labels,num_domains,seg_numdomians_ratio,distance_matrix)


        if(len(set(labels)) < num_domains):
            return 'error'

        hydphob = graph.vs['hydphob']

        core_res_idx = [i for i in range(len(hydphob)) if hydphob[i] > 3]

        labels_core = [labels[i] for i in core_res_idx]

        core_dist_matrix = distance_matrix[core_res_idx,:][:,core_res_idx]

        sil_score = silhouette_score(core_dist_matrix, labels=labels_core)


        for label in set(labels):
            if np.count_nonzero(labels == label) < min_domain_size:
                return 'error'

        return labels,labels, sil_score

    except:
        return 'error'



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
  -clustering [METHOD]      The clustering method
  -numdomains [NUMBER]      The number of domains
  -minsegsize [SIZE]        Minimum segment size
  -mindomainsize [SIZE]     Minimum domain size
  -maxalphahelix [SIZE]     Maximum size of alpha-helix to contract
  -maxsegdomratio [RATIO]   Maximum ratio of segment count to domain count
  -kernel [TYPE]            The type of graph node kernel (**)
  -dispall                  Display all candidate partitionings
  -diffparamx [VALUE]       Diffusion parameter X (***)
  -diffparamy [VALUE]       Diffusion parameter Y (***)
 
  *
  These arguments are necessary
 
  **
  Type should be choosen from:
   lap-exp-diff
   markov-diff
   reg-lap-diff
   markov-exp-diff

  ***
  The parameters diffparamx and diffparamy are coefficient
  (x) and exponent (y) of node count (n) respectively, which
  determine the diffusion parameter, t, for each kernel. (t=xn^y)
"""

def run(argv):

    pdb_file_path = ''
    chain_id = ''
    num_domains = None
    min_seg_size = 25
    max_alpha_helix_size_to_contract = 30
    seg_numdomians_ratio = 1.6
    min_domain_size = 27
    kernel = 'markov-diff'
    display_all_partiotionings = False
    diff_param_x = None
    diff_param_y = None
    dssp_path = ''
    clustering_method = 'spectral'

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
            elif argv[i] == '-clustering':
                clustering_method = argv[i + 1]



    if pdb_file_path=='':
        print('Error: The argument -pdb is necessary')
        argument_error = True
    if chain_id=='':
        print('Error: The argument -chainid is necessary')
        argument_error = True
    if dssp_path=='':
        print('Error: The argument -dssppath is necessary')
        argument_error = True

    if clustering_method not in {'spectral', 'kernel-kmeans'}:
        print('Error: Invalid argument value for -clustering')
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

    # main_graph = graph.copy()

    # contract_alpha_helices(graph, alpha_helices, max_alpha_helix_size_to_contract)

    residue_vartex_map = [None] * n

    for i in range(len(graph.vs)):
        for x in graph.vs[i]['name']:
            residue_vartex_map[x] = i


    num_vtx = len(graph.vs)

    if diff_param_x==None and diff_param_y==None:
        if kernel == 'lap-exp-diff':
            if clustering_method == 'spectral':
                diff_param_x = 0.1105
            else:
                diff_param_x = 0.112
        elif kernel == 'markov-diff':
            if clustering_method == 'spectral':
                diff_param_x = 0.4024
            else:
                diff_param_x = 0.409
        elif kernel == 'reg-lap-diff':
            if clustering_method == 'spectral':
                diff_param_x = 0.035
            else:
                diff_param_x = 0.059
        elif kernel == 'markov-exp-diff':
            if clustering_method == 'spectral':
                diff_param_x = 0.61
            else:
                diff_param_x = 0.66

        diff_param_y = 1


    diff_param = diff_param_x * (num_vtx ** diff_param_y)


    if kernel =='lap-exp-diff':
        kernel_matrix = lap_exp_diff_kernel(graph, diff_param)
    elif kernel == 'markov-diff':
        kernel_matrix = markov_diff_kernel(graph, diff_param)
    elif kernel=='reg-lap-diff':
        kernel_matrix = reg_lap_kernel(graph, diff_param)
    elif kernel == 'markov-exp-diff':
        kernel_matrix = markov_exp_diff_kernel(graph, diff_param)

    if np.isinf(kernel_matrix).any():
        print("Error: Too large diffusion parameter")
        return "kernel infinity"


    distance_matrix = convert_kernel_to_distance(kernel_matrix, method='norm')


    if num_domains == None:

        num_domains = 1
        single_domain_labeling = [0] * n

        labelings = [single_domain_labeling]

        all_assignments = []

        all_assignments.append((1, conv_to_text(
            get_domains_as_segments_by_resnum(get_domains_as_segments(single_domain_labeling), aminoacid_resnums), delimiter=''),
                                '----'))

        opt_num_domains = 1

        smc = SingleMultiClassifier()
        smc_res = smc.predict(graph, acc, hydphob, aminoacid_ca_coords, n)

        # if single_domain_probability <= 0.5 or display_all_partiotionings:
        if smc_res == 'M' or display_all_partiotionings:

            max_sil_score = -1

            while True:
                num_domains += 1
                # result = cluster(num_domains,main_graph, graph,aminoacid_resnums,diff_kernel,min_seg_size,aminoacid_ca_coords, expanded_kernel, seg_numdomians_ratio, sec_struc_labels, residue_vartex_map,aminoacid_letters,acc, beta_strands, max_non_splitted_strand_size, distance_matrix,min_domain_size)
                result = cluster(num_domains, graph, aminoacid_resnums, kernel_matrix, min_seg_size,
                                 seg_numdomians_ratio, distance_matrix, min_domain_size, clustering_method, alpha_helices,max_alpha_helix_size_to_contract)

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
        # if single_domain_probability > 0.5:
        if smc_res == 'S':
            table.add_row(all_assignments[0])

        for assignment in all_assignments[1:]:
            # table.add_row(assignment)
            table.add_row(assignment)

        # table.add_row(all_assignments[0])
        # if single_domain_probability <= 0.5:
        if smc_res == 'M':
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
        result = cluster(num_domains, graph, aminoacid_resnums, kernel_matrix, min_seg_size,
                                seg_numdomians_ratio, distance_matrix, min_domain_size, clustering_method, alpha_helices,max_alpha_helix_size_to_contract)
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

