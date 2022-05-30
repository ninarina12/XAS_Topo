import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix

from mendeleev.fetch import fetch_table
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.utils import class_weight

from plot_imports import *
from periodic_table import fetch_table


def get_class(x, num_classes):
    if num_classes == 2:
        if x > 2: c = 0 # Trivial
        else: c = 1 # Topological
    if num_classes == 3:
        if x > 2: c = 0 # Trivial
        elif x < 2: c = 1 # TI
        else: c = 2 # TSM
    return c


def get_species(data, get_counts=False):
    # get number of distinct species
    species = []
    for entry in data.itertuples():
        species += list(entry.spectra.keys())
    elem_counts = Counter(species)
    species = sorted(list(set(species)))
    print('distinct atomic species:', len(species))
    if get_counts:
        return species, elem_counts
    else:
        return species

    
def load_data(data_path, num_classes=2):
    # read in materials dataframe
    data = pd.read_csv(data_path)
    print('number of samples:', len(data))
    
    # format spectra dictionaries
    data['spectra'] = data['spectra'].apply(eval)
    data['spectra_abs'] = data['spectra_abs'].apply(eval)
    data['spectra_fs'] = data['spectra_fs'].apply(eval)
    data['spectra_abs_fs'] = data['spectra_abs_fs'].apply(eval)
    
    # format structure
    data['structure'] = data['structure'].apply(eval).map(lambda x: Structure.from_dict(x))
    
    # set the binary classes
    data['class_true'] = data['class'].map(lambda x: get_class(x, num_classes))

    # separate out all elements in each sample
    data['elements'] = data['spectra'].map(lambda x: list(x.keys()))
    
    # exclude entries with heavily underrepresented elements
    bad = ['Ar', 'He', 'Kr', 'Ne', 'Pa', 'Xe']
    data = data[~data['elements'].map(lambda x: any(k for k in x if k in bad))]
    data.reset_index(drop=True, inplace=True)

    # get all species
    species = get_species(data)

    return data, species


def build_data(data, species, idx, spec_col='spectra_abs'):
    Z = [Element(specie).Z for specie in species]
    height = len(list(data.iloc[0]['spectra'].values())[0]['x'])
    width = max(Z)

    # initialize array
    xdata = np.zeros((len(idx), height, width))            # element-encoded spectra
    ydata = np.zeros((len(idx), 1))                        # class
    edata = np.zeros((len(idx), width), dtype=int)         # one-hot encoded elements

    # build xdata
    for i, entry in enumerate(data.loc[idx].itertuples()):
        ydata[i,:] = entry.class_true
        xas_dict = eval('entry.' + spec_col)
        struct = entry.structure
        for j, site in enumerate(struct):
            xdata[i,:,site.specie.Z - 1] = xas_dict[str(site.specie)]['y']
            edata[i,site.specie.Z - 1] = 1

    # define type encoding
    type_encoding = ['' for _ in range(width)]
    for z in range(1, width + 1):
        specie = Element.from_Z(z)
        type_encoding[z-1] = specie.symbol

    return xdata, ydata, edata, type_encoding


class standardXAS:
    def __init__(self, ne):
        self.ne = ne
        self.mu = np.zeros(ne)
        self.s = np.ones(ne)
    
    def fit(self, x, e):
        for elem in range(self.ne):
            mask = e[:,elem].ravel() == 1
            if np.sum(e[:,elem]) > 1:
                mu = np.mean(x[mask,:,elem], axis=1)
                s = np.std(x[mask,:,elem], axis=1)
                self.mu[elem] = np.mean(mu)
                self.s[elem] = np.mean(s)
            elif np.sum(e[:,elem]) == 1: # only one sample exists
                self.mu[elem] = np.mean(x[mask,:,elem])
                self.s[elem] = 1.
            else: # no samples exist
                self.mu[elem] = 0.
                self.s[elem] = 1.
                
    def transform(self, x, e):
        for elem in range(self.ne):
            mask = e[:,elem].ravel() == 1
            x[mask,:,elem] -= self.mu[elem]
            x[mask,:,elem] /= self.s[elem]

    def fit_transform(self, x, e):
        self.fit(x,e)
        self.transform(x,e)
        

class clusterXAS:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)

    def remove_outliers(self, x, y, frac, min_size=4):
        mu = np.mean(x, axis=0)
        d = np.sqrt(np.sum(np.square(x - mu),axis=1))
        idx = np.argsort(d)
        n = len(idx); f = 1.
        while n > min_size and f > frac:
            n -= 1
            f = n/float(len(x))
        xf = np.copy(x[idx[:n]])
        yf = np.copy(y[idx[:n]])
        return xf, yf
    
    def _cluster_matching_fit(self, y):
        if len(np.unique(y)) > 1:
            class_weights = dict(enumerate(class_weight.compute_class_weight(
                            'balanced',classes=np.unique(y),y=y)))
        else:
            class_weights = {0:1.,1:1.}
        yp = self.kmeans.labels_
        cmat = contingency_matrix(y, yp).astype(np.float)
        
        # apply class weights:
        for i in range(len(cmat)):
            cmat[i] *= class_weights[i]
            row, col = linear_sum_assignment(-cmat)
        self.cmat = cmat
        self.row = row
        self.col = col
        
    def _cluster_matching_predict(self, x):
        order = [0,1]
        y = self.kmeans.predict(x)
        n = len(y)
        c = np.zeros(n)
        for i in range(n):
            c[i] = order[self.col[order.index(y[i])]]
        return c

    def fit(self, x, y):
        self.kmeans.fit(x)
        self._cluster_matching_fit(y)

    def predict(self, x):
        return self._cluster_matching_predict(x)
    
    
def train_valid_test_split(data, species, valid_size=0.15, test_size=0.15, seed=12):
    ''' data - dataframe of materials examples
        species - list of distinct atomic species
        seed - seed for random train/valid/test split
    '''
    # get element statistics
    ptable, stats = get_element_statistics(data, species)
    print('element count (min):', ptable['count'].min(), '(max):', ptable['count'].max())
    
    def roundup(x):
        return x if x%100 == 0 else x + 100 - x%100
    
    # save periodic table of element representation
    outfile = 'images/element_representation'
    cnorm = (1, roundup(int(ptable['count'].max())))
    cnorm_ratio = (0.1,10.)
    print('cnorm:', cnorm)
    periodic_plot(ptable, attribute='count', colorby='attribute', cmap=cmap,
                  lognorm=True, cnorm=cnorm, output=outfile)
    
    # save periodic table colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=colors.LogNorm(vmin=cnorm[0], vmax=cnorm[1]))    
    sm.set_array([])
    fig, ax = plt.subplots(1,1, figsize=(6,0.4))
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    fontsize = 16
    format_axis(cbar.ax, xlabel='', ylabel='', prop=prop)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('counts', fontsize=fontsize)
    fig.savefig('images/element_representation_cbar.svg', bbox_inches='tight')
    
    # save periodic table ratios colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap_div2, norm=colors.LogNorm(vmin=cnorm_ratio[0], vmax=cnorm_ratio[1]))    
    sm.set_array([])
    fig, ax = plt.subplots(1,1, figsize=(6,0.4))
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal', ticks=(cnorm_ratio[0], 1, cnorm_ratio[1]))
    fontsize = 16
    format_axis(cbar.ax, xlabel='', ylabel='', prop=prop)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_xticklabels([str(cnorm_ratio[0]) + '\ntrivial', 1, str(int(cnorm_ratio[1])) + '\ntopological'])
    ax.set_xlabel('ratio', fontsize=fontsize)
    fig.savefig('images/element_representation_ratio_cbar.svg', bbox_inches='tight')
    
    # save periodic table of trivial element representation
    outfile = 'images/element_representation_triv'
    periodic_plot(ptable, attribute='count_triv', colorby='attribute', cmap=cmap,
                  lognorm=True, cnorm=cnorm, output=outfile)
    
    # save periodic table of topological element representation
    outfile = 'images/element_representation_topo'
    periodic_plot(ptable, attribute='count_topo', colorby='attribute', cmap=cmap,
                  lognorm=True, cnorm=cnorm, output=outfile)
    
    # save periodic table of topological element representation
    outfile = 'images/element_representation_ratio'
    periodic_plot(ptable, attribute='count_ratio', colorby='attribute', cmap=cmap_div2,
                  lognorm=True, cnorm=cnorm_ratio, output=outfile)

    # train/valid/test split data so class representation is balanced for each element
    dev_size = valid_size + test_size
    idx_train, idx_dev = split_data(stats, test_size=dev_size, seed=seed)
    _, stats_dev = get_element_statistics(data.iloc[idx_dev], species)
    idx_test, idx_valid = split_data(stats_dev, test_size=test_size/dev_size, seed=seed)
    
    # add unallocated samples (if any) to training set
    idx_train += data[~data.index.isin(idx_train + idx_valid + idx_test)].index.tolist()
    
    # check for overlap
    print('train/valid/test overlap:', list(set(idx_train) & set(idx_valid) & set(idx_test)))
    print('samples allocated:', len(idx_train + idx_valid + idx_test), '/', len(data))
    
    # save indices
    with open('data/idx_train.txt', 'w') as f:
        for i in idx_train:
            f.write(str(i) + '\n')
            
    with open('data/idx_valid.txt', 'w') as f:
        for i in idx_valid:
            f.write(str(i) + '\n')
            
    with open('data/idx_test.txt', 'w') as f:
        for i in idx_test:
            f.write(str(i) + '\n')

    # populate dataframe with dictionary of classes within train/valid/test sets for each element
    stats['train'] = stats['data'].map(lambda x: class_representation(x, np.sort(idx_train)))
    stats['valid'] = stats['data'].map(lambda x: class_representation(x, np.sort(idx_valid)))
    stats['test'] = stats['data'].map(lambda x: class_representation(x, np.sort(idx_test)))
    
    # bar plot of class representation
    fig, ax = plt.subplots(3,1, figsize=(36,24))
    prop.set_size(32)

    stats = stats.sort_values('symbol')
    for i, dataset in enumerate(['train', 'valid', 'test']):
        split_subplot(ax[0], stats[:len(stats)//3], species[:len(stats)//3], dataset, prop, legend=True)
        split_subplot(ax[1], stats[len(stats)//3:2*len(stats)//3], species[len(stats)//3:2*len(stats)//3], dataset, prop)
        split_subplot(ax[2], stats[2*len(stats)//3:], species[2*len(stats)//3:], dataset, prop)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    fig.savefig('images/train_valid_test_split.svg', bbox_inches='tight')

    return idx_train, idx_valid, idx_test


def get_element_statistics(data, species):    
    ''' data - dataframe of materials examples
        species - list of distinct atomic species
    '''
    # create dictionary indexed by element names storing (index of example containing given element, class of example)
    # class = 1 (TI), 2 (TSM), 3 (Trivial)
    num_classes = 3
    species_dict = {k: [] for k in species}
    for entry in data.itertuples():
        elements = list(entry.spectra.keys())
        for specie in elements:
            species_dict[specie] += [(entry.Index, entry[data.columns.get_loc('class') + 1])]

    # create dataframe to save periodic table of element representation
    ptable = fetch_table('elements')
    ptable['count'] = np.empty((len(ptable)))*np.nan
    ptable['count_topo'] = np.empty((len(ptable)))*np.nan
    ptable['count_triv'] = np.empty((len(ptable)))*np.nan
    ptable['count_ratio'] = np.empty((len(ptable)))*np.nan
    for specie in species:
        ptable.loc[ptable['symbol'] == specie, 'count'] = len(species_dict[specie])
        n_topo = len([k for k in species_dict[specie] if k[1] < num_classes])
        if n_topo == 0: n_topo = np.nan
        n_triv = len([k for k in species_dict[specie] if k[1] == num_classes])
        ptable.loc[ptable['symbol'] == specie, 'count_topo'] = n_topo
        ptable.loc[ptable['symbol'] == specie, 'count_triv'] = n_triv
        ptable.loc[ptable['symbol'] == specie, 'count_ratio'] = float(n_topo)/float(n_triv)

    # create dataframe of element statistics
    stats = ptable[['symbol', 'count']].dropna().reset_index(drop=True)
    stats['data'] = stats['count'].astype('object')
    for specie in species:
        stats.at[stats.index[stats['symbol'] == specie].values[0], 'data'] = species_dict[specie]
    
    return ptable, stats


def split_data(df, test_size, seed):
    ''' df - dataframe of element statistics
        test_size - fraction of total data to set aside for testing
        seed - seed for random number generator to vary splits
    '''
    # initialize output arrays
    idx_train, idx_test = [], []
    
    # remove empty examples
    df = df[df['data'].str.len()>0]

    # sort df in order of fewest to most examples
    df = df.sort_values('count')
    
    for _, entry in df.iterrows():
        df_specie = entry.to_frame().T.explode('data')
        df_specie['id'] = df_specie['data'].map(lambda x: x[0])
        df_specie['class'] = df_specie['data'].map(lambda x: x[1])

        try:
            idx_train_s, idx_test_s = train_test_split(df_specie['id'].values, test_size=test_size,
                                                       random_state=seed, stratify=df_specie['class'].values)
        except:
            # too few examples to perform split - these examples will be assigned based on other constituent elements
            # (assuming not elemental examples)
            pass

        else:
            # add new examples that do not exist in previous lists
            idx_train += [k for k in idx_train_s if k not in idx_train + idx_test]
            idx_test += [k for k in idx_test_s if k not in idx_train + idx_test]
    
    return idx_train, idx_test


def class_representation(x, idx):
    # populate dataframe with dictionary of classes within train/valid/test sets for each element
    num_classes = 3
    d = dict.fromkeys(range(1, num_classes + 1))
    for i in range(1, num_classes + 1):
        d[i] = len([k for k in x if (k[1] == i) & (k[0] in idx)])
    return d


def split_subplot(ax, df, species, dataset, prop, legend=False):
    # bar plot properties
    width = 0.25
    palette = {'train': '#6A71B4', 'valid': '#BB6D89', 'test': '#F7B744'}
    offset = {'train': -width, 'valid': 0, 'test': width}
    
    bx = np.arange(len(species))
    bottom = 0
    num_classes = 3
    for i in range(1,num_classes+1):
        if i == 1: label = dataset
        else: label = None
        
        y = df[dataset].map(lambda x: x[i]/(sum(x.values()) + int(sum(x.values())==0)))
        ax.bar(bx + offset[dataset], y, width, alpha=1. - (i - 1.)/num_classes, bottom=bottom, label=label,
               fc=palette[dataset])
        ax.bar(bx + offset[dataset], y, width, bottom=bottom, fc='none', ec='black', lw=1.5)
        bottom += y
        
    ax.set_xticks(bx)
    ax.set_xticklabels(species)
    ax.set_ylim([0,1.19])

    if dataset == 'train':
        ax.text(0.03, 0.65, 'Trivial', rotation='vertical', color='darkgray', ha='center', va='center',
                fontproperties=prop, transform=ax.transAxes)
        ax.text(0.03, 0.4, 'TSM', rotation='vertical', color='dimgray', ha='center', va='center',
                fontproperties=prop, transform=ax.transAxes)
        ax.text(0.03, 0.15, 'TI', rotation='vertical', color='black', ha='center', va='center',
                fontproperties=prop, transform=ax.transAxes)

    format_axis(ax, '', 'fraction', prop, title=None)
    if legend: ax.legend(frameon=False, prop=prop, ncol=len(palette), loc='upper left')


def get_roc(y_pred, y_true):
    fpr, tpr, th = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, th


def get_optimal_threshold(fpr, tpr, th):
    k = np.argmin((fpr - tpr) + np.abs(tpr - (1. - fpr)))
    print('tpr:', tpr[k], 'fpr:', fpr[k], 'th:', th[k])
    return tpr[k], fpr[k], th[k]


def plot_roc(ax, fpr, tpr, roc_auc, op=None):
    if op: ax.scatter(op[1], op[0], s=48, facecolor='white', edgecolor='black',
                      lw=1.5, zorder=10, label=r'$t_{cutoff}$:' + ' %0.2f' % op[2])

    palette = {'train': '#6A71B4', 'valid': '#BB6D89', 'test': '#F7B744'}

    for i, (label, color) in enumerate(palette.items()):
        ax.plot(fpr[i], tpr[i], color=color, label=r'AUC$_{' + label + r'}$:' + ' %0.2f' % roc_auc[i])
        ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    format_axis(ax, 'false positive rate', 'true positive rate', prop, legend=True)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1.05])
    
    
def plot_precision_recall_fscore(ax, true, pred, num_classes, prop):
    p, r, f1, _ = precision_recall_fscore_support(true, pred, labels=range(num_classes))
    print('metric average (weighted):', precision_recall_fscore_support(true, pred, average='weighted',
                                                                        labels=range(num_classes)))
    
    print('recall:', r)
    print('precision:', p)
    print('f1:', f1)
    
    prop.set_size(16)
    tprop = prop.copy()
    tprop.set_size(14)
    width = 0.6
    ticklabels = ['trivial', 'topological']
    colors = ['#6A71B4', '#F7B744', '#BB6D89']
    cols = [colors[k] for k in range(num_classes)]
    
    ax[0].bar(range(num_classes), height=r, width=width, color=cols, alpha=0.9)
    ax[1].bar(range(num_classes), height=p, width=width, color=cols, alpha=0.9)
    ax[2].bar(range(num_classes), height=f1, width=width, color=cols, alpha=0.9)

    for i in range(len(ax)):
        ax[i].set_ylim([0, 1.1])
        ax[i].set_xticks(range(len(ticklabels)))
        ax[i].set_xticklabels(ticklabels)

    for i in range(num_classes):
        ax[0].text(i, r[i] - 0.1, '{:.2f}'.format(r[i]), color='white', ha='center', fontproperties=tprop)
        ax[1].text(i, p[i] - 0.1, '{:.2f}'.format(p[i]), color='white', ha='center', fontproperties=tprop)
        ax[2].text(i, f1[i] - 0.1, '{:.2f}'.format(f1[i]), color='white', ha='center', fontproperties=tprop)

    format_axis(ax[0], 'class', 'recall', prop)
    format_axis(ax[1], 'class', 'precision', prop)
    format_axis(ax[2], 'class', '$F_1$', prop)


def get_element_results(data, species):    
    ''' data - dataframe of materials examples from a specific dataset
        species - list of distinct atomic species
    '''
    # create dictionary indexed by element names storing (index of example containing given element, class of example)
    species_dict = {k: [] for k in species}
    classes = np.unique(data['class_true'].tolist())
    for entry in data.itertuples():
        struct = entry.structure
        elements = list(set(map(str, struct.species)))
        for specie in elements:
            species_dict[specie] += [(entry.class_true, entry.class_pred)]
    
    # create dataframe to save periodic table of element representation
    ptable = fetch_table('elements')
    ptable['recall_topo'] = np.empty((len(ptable)))*np.nan
    ptable['precision_topo'] = np.empty((len(ptable)))*np.nan
    ptable['f1_topo'] = np.empty((len(ptable)))*np.nan
    ptable['recall_triv'] = np.empty((len(ptable)))*np.nan
    ptable['precision_triv'] = np.empty((len(ptable)))*np.nan
    ptable['f1_triv'] = np.empty((len(ptable)))*np.nan
    for specie in species:
        p, r, f1, _ = precision_recall_fscore_support([k[0] for k in species_dict[specie]],
                                                      [k[1] for k in species_dict[specie]], labels=classes,
                                                      zero_division=0)
        if np.all(np.array([k[0] for k in species_dict[specie]]) == 0):
            p[p==0] = np.nan
            r[r==0] = np.nan
            f1[f1==0] = np.nan
        ptable.loc[ptable['symbol'] == specie, 'recall_topo'] = r[1]
        ptable.loc[ptable['symbol'] == specie, 'precision_topo'] = p[1]
        ptable.loc[ptable['symbol'] == specie, 'f1_topo'] = f1[1]
        ptable.loc[ptable['symbol'] == specie, 'recall_triv'] = r[0]
        ptable.loc[ptable['symbol'] == specie, 'precision_triv'] = p[0]
        ptable.loc[ptable['symbol'] == specie, 'f1_triv'] = f1[0]
    
    return ptable