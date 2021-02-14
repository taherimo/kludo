import os, pickle, math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

here = os.path.dirname(__file__)

class SingleMultiClassifier():

    def __init__(self):
        f = open(os.path.join(here, 'single_multi_classifier.sav'), 'rb')
        self.weak_classifiers = []
        #for i in range(100):
        #self.weak_classifiers.append(pickle.load(f))

        while True:
            try:
                self.weak_classifiers.append(pickle.load(f))
            except EOFError:
                break
        f.close()


    def predict(self, graph, acc, hydphob,aminoacid_ca_coords, n):
        degrees = graph.strength(weights='weight')

        mean_degree = np.mean(degrees)
        degree_var = np.var(degrees)
        closeness = graph.closeness(weights='weight')
        betweenness = graph.betweenness(weights='weight')

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


        feature_vec = np.array([len(graph.es) / n, n, mean_degree, degree_var, degrees_hydphob_corr,
                 degree_acc_corr, closeness_hydphob_corr, closeness_acc_corr, betweenness_hydphob_corr,
                 betweenness_acc_corr,num_cores,core_hydph_acc_expl_var[0],core_hydph_acc_expl_var[1],
                 core_hydph_acc_expl_var[2]]).reshape(1, -1)


        predictions = []
        for clf in self.weak_classifiers:
            predict = clf.predict_proba(feature_vec)
            # predict_train = clf.predict_proba(X)
            predictions.append([p[1] for p in predict])

        predictions = np.array(predictions)
        single_domain_probability = np.mean(predictions, 0)

        if single_domain_probability <= 0.5:
            return 'M'
        else:
            return 'S'

