import Bio.PDB
import numpy as np

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
