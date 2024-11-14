import numpy as np
import copy

from scipy.stats import binom
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC

import dio3

pars_dict = {'RF':{'max_depth': 15, 'n_estimators': 50,},
             'NN_SK':{'activation': 'relu', 'hidden_layer_sizes': (20, 10,),},
             'BDT':{'max_depth': 15},
             'LR':{'max_iter':1000},
             'SVC':{'C':1, 'gamma':2, 'probability':True},
             'NN': {'optimizer': 'adam', 'learning_rate': 0.001, 'epochs': 500, 'activation': 'tanh', 
                       'hidden_layer_sizes': [20, 10], 
                       'nfeatures': 0, 'classes': None, 'verbose': 0,
                       'drop_rate': 0.0, 'batch_size': 200,
                      'validation_split': None}
            }

def logpar_index(E, alpha, beta, E0):
    return alpha + 2. * beta * np.log(E / E0)

def get_log10_Epeak(alpha, beta, E0):
    return (np.log(E0) + (2. - alpha) / (2. * beta)) / np.log(10)

def logpar_index_cat(E, table):
    alpha = table['LP_Index']
    beta = table['LP_beta']
    E0 = table['Pivot_Energy']
    return logpar_index(E, alpha, beta, E0)

def logpar_Epeak_cat(table):
    alpha = table['LP_Index']
    beta = table['LP_beta']
    E0 = table['Pivot_Energy']
    return get_log10_Epeak(alpha, beta, E0)


def get_classifier(alg, par_dict):
    if alg == 'RF':
        return RandomForestClassifier(max_features="sqrt", **par_dict)
    elif alg == 'BDT':
        return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(**par_dict), 
                                  n_estimators=50)
    elif alg == 'NN_SK':
        return MLPClassifier(alpha=1.e-5, max_iter=1000, **par_dict)
    elif alg == 'LR':
        return LogisticRegression(C=1e5, **par_dict)
    elif alg == 'NN':
        return NNClassifierTF(**par_dict)
    elif alg == 'SVC':
        return SVC(**par_dict)


class NNClassifierTF:
    def __init__(self, **par_dict):
        import tensorflow as tf
        activation = par_dict.get('activation', 'tanh')
        hidden_layer_sizes = par_dict.get('hidden_layer_sizes', [20, 10])
        nfeatures = par_dict.get('nfeatures', 10)
        self.classes_ = par_dict.get('classes')
        optimizer_name = par_dict.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            learning_rate = par_dict.get('learning_rate', 0.001)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.batch_size = par_dict.get('batch_size', 200)
        self.epochs = par_dict.get('epochs', 200)
        self.drop_rate = par_dict.get('drop_rate', 0.3)
        self.validation_split = par_dict.get('validation_split')
        self.verbose = par_dict.get('verbose', 1)
        nclasses = len(self.classes_)
        self.class_dict = {self.classes_[i]:i for i in range(nclasses)}
        l2 = par_dict.get('l2', 0.001)
        l2_reg = tf.keras.regularizers.L2(l2=l2)
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(nfeatures,)))
        for nh in hidden_layer_sizes:
            model.add(tf.keras.layers.Dense(nh, activation=activation, kernel_regularizer=l2_reg))
            model.add(tf.keras.layers.Dropout(self.drop_rate))
        model.add(tf.keras.layers.Dense(nclasses, activation='softmax'))
        self.accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model = model
        
    def fit(self, X_train, y_train, **kwargs):
        y_train_int = np.array([self.class_dict[cl] for cl in y_train])
        self.model.fit(X_train, y_train_int,  
                       batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                       validation_split=self.validation_split,
                       **kwargs)
    
    def test_accuracy(self, X_test, y_test, **kwargs):
        y_test_int = np.array([self.class_dict[cl] for cl in y_test])
        y_pred = self.predict_proba(X_test)
        self.accuracy.update_state(y_test_int, y_pred, **kwargs)
        return self.accuracy.result().numpy()

    def predict_proba(self, X_test):
        return self.model.predict(X_test, verbose=self.verbose)
    
    
    
def get_cl_mask(cls, xdf, cname='CLASS1'):
    """
        Find FGL sources which belong to a list a classes
        Input:
            cls - list of FGL classes
            xdf - dataframe with FGL sources
        Output:
            mask - bool array which selects the sources
    """
    mask = False
    for cl in cls:
        mask |= xdf[cname] == cl
    return mask

par_dict_gmm = dict(n_components=2, init_params='random', n_init=3, random_state=4)
par_dict_rf = {'max_depth': 15, 'n_estimators': 50, 'random_state': 4}
def get_model(sep_model='GMM', par_dict=None):
    if sep_model == 'GMM':
        if par_dict is None:
            par_dict = par_dict_gmm
        return GaussianMixture(**par_dict)
    elif sep_model == 'RF':
        if par_dict is None:
            par_dict = par_dict_rf
        return RandomForestClassifier(max_features="sqrt", **par_dict)
   

def get_cats(pred, thres=0.5, out_thres=False):
    """
        Input:
            pred - dictionary {fgl class: nsrc-vector with probabilities to belong to class "1"}
    """
    res_dict = {}
    # determine the threshold to separate classes into two categories
    mean_vs = np.array([np.mean(v) for v in pred.values()])
    mask = np.zeros_like(mean_vs, dtype=bool)
    if 0:
        # hack to remove the first option for now
        # option 1: compare mean probabilities with input threshold
        mask = mean_vs > thres
    if sum(mask) == 0 or sum(~mask) == 0:
        # option 2:
        # if all or no sources fall in class "1"
        # calculate the threshold by averaging all probabilities
        all_vs = []
        for value in pred.values():
            all_vs.extend(list(value))
        thres = np.mean(all_vs)
    mask = mean_vs > thres
    if sum(mask) == 0 or sum(~mask) == 0:
        # option 3:
        # if all or no sources fall in class "1"
        # calculate the threshold by averaging mean probabilities
        thres = np.mean(mean_vs)
        
    # split the classes of sources
    for cl, value in pred.items():
        res_dict[cl] = int(np.mean(value) > thres)
        
    if out_thres:
        return res_dict, thres
    else:
        return res_dict

def get_tree_cats(cat_dict, cls, xdf, cname='CLASS1',
                  features=[], nmin=20, max_depth=3, 
                  nsrc_dict=None, mdl_dict=None,
                 sep_model='GMM', par_dict=None,
                 nmin_GMM=10):
    # find the sources which belong to classes in "cls" list
    if nsrc_dict is None:
        nsrc_dict = get_nsrc_dict(xdf, classes='associated')
    cls_mask = get_cl_mask(cls, xdf, cname=cname)
    ns = len(xdf)
    # check that there are sufficiently many classes and sources
    if len(cls) > 1 and sum(cls_mask) > nmin:
        # calculate probabilities for a source to belong to class "1"
        mdl = get_model(sep_model=sep_model, par_dict=par_dict)
        if sep_model == 'GMM':
            if nmin_GMM is None:
                cls_mask_fit = cls_mask
            else:
                cls_loc = [cl for cl in cls if nsrc_dict[cl] >= nmin_GMM]
                print(cls_loc)
                cls_mask_fit = get_cl_mask(cls_loc, xdf, cname=cname)
            mdl.fit(xdf[features][cls_mask_fit])
        elif sep_model == 'RF':
            # find two largest classes
            # probably needs testing
            nsrc_df = nsrc_dict2df(nsrc_dict).loc[cls]
            nsrc_df.sort_values('n', ascending=False, inplace=True)
            cls_loc = list(nsrc_df.index[:2])
            if 0:
                nsrc_loc = np.sort([sum(xdf[cname] == cl) for cl in cls])[::-1]
                thres = nsrc_loc[1] - 0.5
                cls_loc = [cl for cl in cls if sum(xdf[cname] == cl) > thres][:2]
            #print('classes, sources:', cls_loc, nsrc_loc)
            # select X and y features for the two classes
            cls_mask_loc = get_cl_mask(cls_loc, xdf, cname=cname)
            X_train = xdf[features][cls_mask_loc]
            y_train = np.array(xdf[cname][cls_mask_loc] == cls_loc[1], dtype=int)
            mdl.fit(X_train, y_train)

        pred = {}
        for cl in cls:
            cl_mask = xdf[cname] == cl
            pred[cl] = mdl.predict_proba(xdf[features][cl_mask]).T[1]
        # get the dictionary that splits the classes into two categores "0" and "1"
        new_cat_dict = get_cats(pred)
        # check the min number of sources in children nodes
        minn_sources = ns
        for j in range(2):
            cls_new = [cl for cl in new_cat_dict.keys() if new_cat_dict[cl] == j]
            if nsrc_dict is not None:
                nsrc = np.sum([nsrc_dict[cl] for cl in cls_new])
            else:
                nsrc = np.sum(get_cl_mask(cls_new, xdf, cname=cname))
            
            minn_sources = min(minn_sources, nsrc)
        # if children have more than minimal number of sources than save and iterate
        if minn_sources > nmin:
            if mdl_dict is not None:
                # save the classification model in the node
                mdl_dict[cat_dict[cls[0]]] = mdl
            for cl in cls:
                #cat_dict[cl].append(new_cat_dict[cl]) # represent nodes as 0-1 lists
                cat_dict[cl] += str(new_cat_dict[cl]) # represent nodes as 0-1 strings
            for j in range(2):
                cls_new = [cl for cl in new_cat_dict.keys() if new_cat_dict[cl] == j]
                # subsplit the node if max depth is not reached (common root has depth 0)
                if len(cat_dict[cls_new[0]]) < max_depth + 1:
                    get_tree_cats(cat_dict, cls_new, xdf, features=features, 
                                  nmin=nmin, max_depth=max_depth, nsrc_dict=nsrc_dict,
                                  mdl_dict=mdl_dict,
                                  sep_model=sep_model, par_dict=par_dict,
                                 nmin_GMM=nmin_GMM)
    return None

def cat_dict2tree_cat(cat_dict):
    """
        Transform dictionary {fgl class:node} to {node: list of fgl classes}
    """
    res = {}
    for cat, node in cat_dict.items():
        if node not in res:
            res[node] = []
        res[node].append(cat)
    return res

def tree2max_tree(tree_cat_dict):
    """
        Add 0's at the end of the node names so that all node names have the same length.
    """
    nmax = max_node_length(tree_cat_dict)
    res = {}
    for node, cls in tree_cat_dict.items():
        for i in range(len(node), nmax):
            node += '0'
        res[node] = cls
    return res

def max_tree2dec_tree(max_tree_cat_dict):
    #return {int(node, 2):cls for node, cls in max_tree_cat_dict.items()}
    return {(i+1):cls for i, cls in enumerate(max_tree_cat_dict.values())}

def dec_tree2dmax_tree(dec_tree_cat_dict):
    max_node = np.max(np.array(list(dec_tree_cat_dict.keys()), dtype=float))
    width = int(np.floor(np.log2(max_node))) + 1
    width += 1 # also add a leading 0
    return {np.binary_repr(node, width=width):cls for node, cls in dec_tree_cat_dict.items()}

def prune_tree_cat(tree_cat0, xdf, prune_alg='RF', max_depth=5, 
                   features=[], pars_dict=pars_dict, nsrc_dict=None, class_name='CLASS1'):
    '''
        return a tree without the node with the minimal number of sources
    '''
    tree_cat = copy.deepcopy(tree_cat0)
    if nsrc_dict is None:
        nsrc_dict = get_nsrc_dict(xdf, classes='associated')
    ns = len(xdf)

    # select the node to prune
    prune = np.inf
    for node, cls in tree_cat.items():
        nsrc = np.sum([nsrc_dict[cl] for cl in cls])
        if nsrc < prune:
            prune = nsrc
            prune_node = node
    
    
    prune_cls = tree_cat.pop(prune_node)
    
    # move classes for the other nodes one level up
    prune_parent = prune_node[:-1]
    other_children = all_leaf_children(tree_cat.keys(), prune_parent)
    for node in other_children:
        new_node = prune_parent + node[len(prune_parent) + 1:]
        tree_cat[new_node] = tree_cat.pop(node)
    
    # redistribute the classes from the removed node to remaining nodes
    #print(prune_cls, prune_node)
    # determine the classification model to distribute the classes in the node
    ys = np.zeros(ns, dtype='<U%i' % (max_depth + 1))
    clf_mask = np.zeros(ns, dtype=bool)
    cls_masks_all = []
    for key, cls in tree_cat.items():
        cls_mask = get_cl_mask(cls, xdf, cname=class_name)
        cls_masks_all.append(cls_mask)
        ys[cls_mask] = key
        clf_mask[cls_mask] = True
    cls_masks_all = np.array(cls_masks_all)

    X = xdf[features][clf_mask]
    y = ys[clf_mask]
    
    clf = get_classifier(prune_alg, pars_dict[prune_alg])
    clf.fit(X, y)
    
    # redistribute the classes
    for cl in prune_cls:
        cl_mask = xdf[class_name] == cl
        X_cl = xdf[features][cl_mask]
        pred_probs = clf.predict_proba(X_cl)
        pred_probs_sum = np.sum(pred_probs, axis=0)
        #find class with maximal sum of probabilities
        ind_max = np.argmax(pred_probs_sum)
        node = clf.classes_[ind_max]
        #print(cl, pred_probs_sum, ind_max, node, tree_cat[node])
        tree_cat[node].append(cl)
    #print(tree_cat)
    return tree_cat


def max_node_length(tree):
    nmax = 0
    for node in tree:
        nmax = max(nmax, len(node))
    return nmax

def is_in_tree(node, tree=None):
    if tree is None:
        return True
    nmax = max_node_length(tree)
    nnode = len(node)
    if nnode > nmax:
        return False
    elif nnode == nmax:
        return node in tree
    else:
        for nd in tree:
            if node == nd[:nnode]:
                return True
    return False
    
def is_internal(node, tree=None):
    if tree is None:
        return True
    daughter0 = node + '0'
    daughter1 = node + '1'
    if is_in_tree(daughter0, tree=tree) or is_in_tree(daughter1, tree=tree):
        return True
    return False

def exist_internal(tree_cat_dict):
    tree = list(tree_cat_dict.keys())
    for node in tree:
        if is_internal(node, tree):
            return True
    return False

def remove_internal(tree_cat_dict):
    tree = list(tree_cat_dict.keys())
    for node in tree:
        if is_internal(node, tree):
            tree_cat_dict.pop(node)
    return None
  
def is_terminal(node, tree=None):
    if tree is None:
        return True
    if node in tree:
        return True
    return False

def get_max_nodes(tree, max_tree=None):
    nmax = max_node_length(tree)
    max_nodes = []
    for i, node in enumerate(tree):
        if len(node) == nmax and is_internal(node, max_tree):
            max_nodes.append(i)
    return max_nodes

def max_tree_length(forest):
    nmax = 0
    for tree in forest:
        nmax = max(nmax, max_node_length(tree))
    return nmax

def get_all_combinations(n):
    return [np.array(list(np.binary_repr(k, width=n)), dtype=int).astype(bool) for k in range(2**n)]

def grow_trees(forest, max_tree=None):
    # split maximal nodes in all trees
    nmax = max_tree_length(forest)
    new_forest = copy.deepcopy(forest)
    for tree in forest:
        if max_node_length(tree) == nmax:
            all_max_nodes = np.array(get_max_nodes(tree, max_tree=max_tree), dtype=int)
            #print(all_max_nodes)
            for mask in get_all_combinations(len(all_max_nodes))[1:]:
                #print(all_max_nodes, mask, all_max_nodes.dtype, mask.dtype)
                max_nodes = all_max_nodes[mask]
                new_tree = copy.deepcopy(tree)
                for i in max_nodes:
                    #print('new_tree', new_tree)
                    split_node = new_tree[i]
                    #print('new_tree', new_tree)
                    #print('split_node', split_node)
                    new_tree.append(split_node + "0")
                    new_tree.append(split_node + "1")
                    #print('new_tree', new_tree)
                new_tree = [new_tree[i] for i in range(len(new_tree)) if i not in max_nodes]
                new_forest.append(new_tree)
    return new_forest

    
def get_forest(max_depth=None, max_tree=None):
    node = "0"
    tree = [node]
    forest = [tree]
    if max_depth is None and max_tree is None:
        return None
    if max_depth is None:
        max_depth = max_node_length(max_tree)
    for depth in range(max_depth):
        forest = grow_trees(forest, max_tree=max_tree)
    return forest

def print_forest(forest, remove_root=True):
    if not remove_root:
        print(forest)
    else:
        for tree in forest:
            ptree = [node[1:] for node in tree]
            if len(ptree) > 0:
                print(ptree)
    return None

def get_horiz_trees(cat_tree, mode='repeat'):
    subtree = ['0']
    horiz_trees = []
    while max_node_length(subtree) < max_node_length(cat_tree):
        subtree = get_children(subtree, cat_tree, mode=mode)
        horiz_trees.append(subtree)
    return horiz_trees

def cat_dict2tree(cat_dict):
    return list(np.sort(list(set(cat_dict.values()))))

def get_children(sub_tree, tree, mode='repeat'):
    res_tree = []
    for node in sub_tree:
        if is_internal(node, tree=tree):
            for child in ['0', '1']:
                if is_in_tree(node + child, tree=tree):
                    res_tree.append(node + child)
        elif mode == 'repeat':
            res_tree.append(node)
        elif mode == 'add_0':
            res_tree.append(node + '0')
    return res_tree

def all_leaf_children(tree, node, include_parents=False):
    res = [nd for nd in tree if nd.startswith(node)]
    if len(res) == 0 and include_parents:
        while node not in tree:
            node = node[:-1]
        res = [node]
    return res

def sub_tree2tree_cat_dict(subtree, tree, tree_cat_dict, include_parents=False):
    res = {}
    for nd in subtree:
        res[nd] = []
        for leaf in all_leaf_children(tree, nd, include_parents=include_parents):
            res[nd].extend(tree_cat_dict[leaf])
    return res


def get_subtree(node, tree):
    return [nd for nd in tree if nd.startswith(node)]


def trim2depth(tree_cat_dict, depth=None):
    tree = tree_cat_dict.keys()
    if depth is None or max_node_length(tree) <= depth + 1:
        return tree_cat_dict
    else:
        tree_cat_dict_res = tree_cat_dict.copy()
        while max_node_length(tree) > depth + 1:
            for node in tree:
                if len(node) > depth + 1:
                    new_node = node[:depth + 1]
                    subtree = get_subtree(new_node, tree)
                    new_classes = []
                    for leaf in subtree:
                        new_classes.extend(tree_cat_dict_res.pop(leaf))
                    tree_cat_dict_res[new_node] = new_classes
                    tree = tree_cat_dict_res.keys()
                    break
    return tree_cat_dict_res

def reduce_probs(parents, children, probs):
    res = []
    for node in parents:
        daugh = all_leaf_children(children, node)
        inds = [children.index(d) for d in daugh]
        res.append(np.sum(probs[inds], axis=0))
    return np.array(res)

def get_reliability(y_true, y_prob, bins, alpha=0.68):
    """
        pgrid - grid of probabilities
        probs - probability for prediction of the class
        labels - 1 if element belongs to the class, 0 otherwise
    """
    res_dict = {}
    nelem = np.histogram(y_prob, bins=bins)[0]
    mask = nelem > 0
    nelem = nelem[mask]
    res_dict['prob_true'] = np.histogram(y_prob, bins=bins, weights=y_true)[0][mask] / nelem
    p = (bins[1:] + bins[:-1])[mask] / 2
    intervals = binom.interval(alpha, nelem, p, loc=0) / nelem
    # [lower error, upper error]
    res_dict['err_true'] = np.array([p - intervals[0], intervals[1] - p])
    
    res_dict['prob_pred'] = np.histogram(y_prob, bins=bins, weights=y_prob)[0][mask] / nelem
    res_dict['nelem'] = nelem
    return res_dict


def cl_list2str(cls):
    cls_str = ''
    for cl in cls:
        cls_str += '%s, ' % cl
    return cls_str[:-2]

def df2tex(df, fmt='%.1f', index=False, index_name=''):
    ff = lambda x: ('%s' % fmt) % x
    tst = df.to_latex(index=index, float_format=ff)
    for st in ['toprule', 'midrule', 'bottomrule']:
        tst = tst.replace(st, 'hline')
    if index:
        tst = tst.replace('{}', index_name)
    return tst

def get_feature_importances(clf, clm_name='Importance', sort=True):
    #data = np.array([clf.feature_names_in_, clf.feature_importances_]).T
    #dff = pd.DataFrame(columns=['Feature', 'Imortance'], data=data)
    data = np.array(clf.feature_importances_)
    index = np.array(clf.feature_names_in_)
    dff = pd.DataFrame(columns=[clm_name], data=data, index=index)
    if sort:
        dff = dff.sort_values(clm_name, ascending=False)
    return dff

def make_thres_bins(thres, values, nbins=100, eps=1.e-15):
    thres_bins = np.linspace(0.-eps, 1.+eps, nbins + 1) # regular grid
    thresc = (thres_bins[1:] + thres_bins[:-1])/2 # centers in the regular grid
    nn = np.histogram(thres, bins=thres_bins)[0] # number of entries in each bin
    mask = (nn != 0.) # mask that shows bins with non-zero entries
    # average value in each bin where there are non-zero number of entries
    res = np.histogram(thres, bins=thres_bins, weights=values)[0][mask] / nn[mask]
    return res, thresc, mask

def nsrc_dict2df(nsrc_dict, sort=True):
    df = pd.DataFrame(index=nsrc_dict.keys(), data=nsrc_dict.values(), columns=['n'])
    if sort:
        df.sort_values('n', ascending=False, inplace=True)
    return df

def get_nsrc_dict(xdf, class_column='CLASS1', classes='associated'):
    if type(classes) == str and classes in ['all', 'associated']:
        all_classes = list(set(xdf[class_column]))
        if classes == 'all':
            classes = all_classes
        elif classes == 'associated':
            classes = [cl for cl in all_classes if not cl.startswith('un')]
    elif type(classes) != list:
        print('classes should be a list or "all" or "associated"')
        return None
    return {cl:np.sum(xdf[class_column] == cl) for cl in classes}

def max_classes(cls, nsrc_dict=None, ncl=3):
    if len(cls) < ncl or nsrc_dict is None:
        return cls
    nmin = 0
    res = []    
    for cl in cls:
        if nsrc_dict[cl] > nmin:
            res.append(cl)
            if len(res) >= ncl:
                nmin = nsrc_dict[res[-ncl]]
    return res[-ncl:]


def list2label(lst, nsrc_dict={}, ncl=3, max_only=True):
    res = ''
    if max_only:
        lst_ = max_classes(lst, nsrc_dict, ncl=ncl)
    else:
        lst_ = lst
    for s in lst_:
        res += '%s ' % s
    if max_only and len(lst) > ncl:
        res += '+ '
    return res


eps = 1.e-30
def get_sample_ratio(mdlu_probs, mdla_probs, binary=False, max_sample=20):
    sample_ratio = mdlu_probs / (mdla_probs + eps) * np.sign(mdla_probs)
    if binary:
        sample_ratio = np.array(sample_ratio > max_sample).astype(float)
    else:
        sample_ratio = np.minimum(sample_ratio, max_sample)
    return sample_ratio

def get_max_class(classes, nsrc_df):
    return nsrc_df.loc[classes].sort_values('n').index[-1]


#def get_config_fn(cat_name='4FGL_DR4', folder='../config/'):
#    fn = '%s_%s.yaml' % (cat_name, descr)
#    return folder + fn

def get_config_dict(cat_name='4FGL-DR4', baseline_case='4FGL-DR4_RF_RF', case='4FGL-DR4_RF_RF', 
                    baseline_fn=None, cases_fn=None, folder='../config/'):
    if baseline_fn is None:
        baseline_fn = folder + '%s_baseline.yaml' % baseline_case
    if cases_fn is None:
        cases_fn = folder + '%s_cases.yaml' % baseline_case
    #cases_fn = folder + '%s_cases.yaml' % cat_name
    cdict = dio3.loaddict(baseline_fn)
    cases_dict = dio3.loaddict(cases_fn)
    cdict.update(cases_dict[baseline_case])
    cdict.update(cases_dict[case])
    cdict['case'] = case
    cdict['baseline_case'] = baseline_case
    return cdict



def get_xdf_fn(cdict):
    cat_name = cdict.get('cat_name', '4FGL-DR4')
    Eind = int(cdict.get('Eind', 1000))
    add_plec_pars = int(cdict.get('add_plec_pars', 0))
    folder_out = cdict.get('root_folder', '../') + cdict.get('folder_out', 'data_out/')
    if add_plec_pars:
        plec_str = '_add_plec_pars'
    else:
        plec_str = ''
    return folder_out + 'xdf_%s_Eind%iMeV%s.csv' % (cat_name, Eind, plec_str)

def get_cat_fn(cdict):
    cat_name = cdict.get('cat_name', '4FGL-DR4')
    version = cdict.get('version')
    folder_in = cdict.get('root_folder', '../') + cdict.get('folder_in', 'data_in/')
    if cat_name == '4FGL-DR4':
        if version is None:
            fn = folder_in + 'gll_psc_v34.fit'
        else:
            fn = folder_in + 'gll_psc_v%s.fit' % version
    elif cat_name == '4FGL-DR3':
        if version is None:
            fn = folder_in + 'gll_psc_v31.fit'
        else:
            fn = folder_in + 'gll_psc_v%s.fit' % version
    elif cat_name == '4FGL-DR2':
        if version is None:
            fn = folder_in + 'gll_psc_v27.fit'
        else:
            fn = folder_in + 'gll_psc_v%s.fit' % version
    elif cat_name == '4FGL-DR1':
        if version is None:
            fn = folder_in + 'gll_psc_v22.fit'
        else:
            fn = folder_in + 'gll_psc_v%s.fit' % version
    elif cat_name == '8year':
        if version is None:
            fn = folder_in + 'gll_psc_8year_v6.fit'
        else:
            fn = folder_in + 'gll_psc_8year_v%s.fit' % version
    elif cat_name == '3FGL':
        if version is None:
            fn = folder_in + 'gll_psc_v16.fit'
        else:
            fn = folder_in + 'gll_psc_v%s.fit' % version
    return fn

descr_keys = ['cat_name', 'sep_model', 'nmin', 'prune', 'assoc_sources', 'features']
def get_description(cdict, lst=descr_keys):
    if cdict['sep_model'] == '2class':
        return '_%s_2class_gal_egal' % cdict['cat_name']
    descr = ''
    for key in lst:
        if key in cdict:
            if key in ['nmin', 'prune', 'sep_model']:
                descr += '_%s_%s' % (key, cdict[key])
            else:
                descr += '_%s' % cdict[key]
    return descr

def get_tree_cat_fn(cdict, lst=descr_keys):
    descr = get_description(cdict, lst=lst)
    folder_out = cdict.get('root_folder', '../') + cdict.get('folder_out', 'data_out/')
    return '%s/tree_cat%s.yaml' % (folder_out, descr)

def get_weights_fn(cdict):
    folder_out = cdict.get('root_folder', '../') + cdict.get('folder_out', 'data_out/')
    cat_name = '4FGL-DR4'
    assoc_sources = cdict.get('assoc_sources', 'all_assoc')
    feature_mode = cdict.get('features', 'all_features')
    nGMM = cdict.get('nGMM_w', 10)
    weights_fn = '%s/weights/weights_%s_%s_%s_%ikernels.csv' % \
    (folder_out, cat_name, assoc_sources, feature_mode, nGMM)
    return weights_fn


def get_alg_descr(cdict, pars_dict=pars_dict):
    alg = cdict.get('class_alg', 'RF')
    sep_model = cdict.get('sep_model', 'GMM')
    cat_name = cdict.get('cat_name', '4FGL-DR4')

    assoc_sources = cdict.get('assoc_sources', 'all_assoc')
    feature_mode = cdict.get('features', 'all_features')
    nmin = cdict.get('nmin')
    prune = cdict.get('prune')
    train_weighting = cdict.get('train_weighting')
    test_weighting = cdict.get('test_weighting')
    test_gal_plane = int(cdict.get('test_gal_plane', 0))

    if nmin is not None: 
        nmin = int(nmin)
        
    # catalog, input and output features
    alg_descr = '%s_%s_%s' % (cat_name, assoc_sources, feature_mode)
    
    # definition of classes
    alg_descr += '_%s' % sep_model
    if sep_model != '2class' and nmin is not None:
        alg_descr += '_nmin%s' % (nmin)
    if prune is not None:
        alg_descr += '_prune%s' % (prune)

    # classification algorithm and training
    alg_descr += '_%s' % alg
    if train_weighting is not None:
        alg_descr += '_%s_training' % train_weighting
    if test_weighting is not None:
        alg_descr += '_%s_test' % test_weighting

    if alg == 'RF':
        alg_descr += '_depth%i' % pars_dict[alg]['max_depth']
    elif alg.startswith('NN'):
        alg_descr += '_layers'
        for ls in pars_dict[alg]['hidden_layer_sizes']:
            alg_descr += '_%i' % ls
    if test_gal_plane:
        glon_max_test = int(cdict.get('glon_max_test', 30))
        glat_max_test = int(cdict.get('glat_max_test', 30))
        alg_descr += '_test_gal_plane_lon%i_lat%i_around_GC' % (glon_max_test, glat_max_test)
    
    return alg_descr


def get_cat_descr(cdict, add_pshift=0):
    sep_model = cdict.get('sep_model', 'GMM')
    cat_name = cdict.get('cat_name', '4FGL-DR4')
    version = cdict.get('version')

    assoc_sources = cdict.get('assoc_sources', 'all_assoc')
    feature_mode = cdict.get('features', 'all_features')
    nmin = cdict.get('nmin')
    train_weighting = cdict.get('train_weighting')
    tree_cat_fn = get_tree_cat_fn(cdict)
    tree_cat_dict = dio3.loaddict(tree_cat_fn)
    nclasses = len(tree_cat_dict)
    
    prior_shift = cdict.get('prior_shift')


    if nmin is not None: 
        nmin = int(nmin)
        
    # catalog, input and output features
    cat_descr = '%s' % (cat_name)
    
    if version is not None:
        cat_descr += '_v%i' % (version)
    cat_descr += '_%iclasses' % (nclasses)
        
    # definition of classes
    cat_descr += '_%s' % sep_model
    #if sep_model != '2class' and nmin is not None:
    #    cat_descr += '_nmin%s' % (nmin)

    if assoc_sources != 'all_assoc':
        cat_descr += '_%s' % assoc_sources
        
    if feature_mode != 'all_features':
        cat_descr += '_%s' % feature_mode
    
    # classification algorithm and training
    if train_weighting is not None:
        cat_descr += '_%s' % train_weighting
    if add_pshift and prior_shift is not None and int(prior_shift) == 1:
        cat_descr += '_%s' % cdict.get('unas_sources', 'unas_unk')
        cat_descr += '_prior_shift_npar%i_%iGauss' % (cdict.get('npar', 1), cdict.get('nGauss', 0))

    #cat_descr += '_prob_cat.csv'
    return cat_descr

def get_pcat_fn(cdict):
    add_pshift = int(cdict.get('prior_shift', 0))
    root_folder = cdict.get('root_folder', '../')
    folder_out = '%sresults/%s_probabilistic_catalogs/' % (root_folder, get_cat_descr(cdict, add_pshift=0))
    fn = '%s%s_%s' % (folder_out, get_cat_descr(cdict, add_pshift=add_pshift), 'prob_cat.csv')
    return fn



def get_cl_ind(dec_tree_cat_dict, cl):
    for ind, cls in dec_tree_cat_dict.items():
        if cl in cls:
            return ind

        
def get_max_classes_dict(tree_cat_dict, nsrc_df):
    res = {}
    for node, classes in tree_cat_dict.items():
        cl = nsrc_df.loc[classes].sort_values('n').index[-1]
        res[node] = cl
    return res
