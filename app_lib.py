import os
import pandas as pd
import numpy as np
import pickle
import re
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn.feature_selection import VarianceThreshold
import calcs
import plots

def does_exist(chembl):
    if chembl in os.listdir("results/"):
        exist = True
    else:
        exist = False
        os.mkdir("results/" + chembl)
        os.mkdir("results/" + chembl + "/files")
        os.mkdir("results/" + chembl + "/models")
        os.mkdir("results/" + chembl + "/figures")

    return exist


def target_search(chembl):
    '''Select chembl ID of target '''

    selected_target = chembl
    activity = new_client.activity
    res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(res)
    df.standard_type.unique()
    df.to_csv('results/' + chembl + '/files/bioactivity_data.csv', index=False)

    # Remove compounds with missing standard value
    df2 = df[df.standard_value.notna()]
    df2 = df2[df.canonical_smiles.notna()]
    df2_nr = df2.drop_duplicates(['canonical_smiles'])

    # SAVE
    selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
    df3 = df2_nr[selection]
    df3.to_csv('results/' + chembl + '/files/preprocessed.csv', index=False)

    return df2_nr

def preprocess(df2, chembl):
    '''Preprocessing data'''

    df4 = pd.read_csv('results/' + chembl + '/files/preprocessed.csv')
    bioactivity_class = []
    for i in df4.standard_value:
        if float(i) >= 10000:
            bioactivity_class.append("inactive")
        elif float(i) <= 1000:
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")

    bioact = pd.DataFrame(bioactivity_class, columns=['bioactivity_class'])
    df5 = pd.concat([df4, bioact], axis=1)
    df5.to_csv('results/' + chembl + '/files/curated.csv', index=False)
    return df5

def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])
        if (i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors

def eda(df_combined, chembl):

    df_norm = calcs.norm_value(df_combined)
    df_final = calcs.pIC50(df_norm)
    # print(df_final.pIC50.describe())
    df_2class = df_final[df_final.bioactivity_class != 'intermediate']
    df_2class.to_csv('results/' + chembl + '/files/pIC50.csv')

    return df_2class

def visual_eda(df_2class, chembl, pp):
    lips = ["pIC50", "MW", "LogP", "NumHDonors", "NumHAcceptors"]
    plots.make_plots(1,0,0,0,df_2class, chembl, pp)
    plots.make_plots(0,1,0,0,df_2class, chembl, pp)
    for var in lips:
        calcs.mannwhitney(var,df_2class, chembl, verbose=False)
        calcs.mannwhitney(var,df_2class, chembl, verbose=False)
        plots.make_plots(0, 0, 1, var, df_2class, chembl, pp)

def padel(chembl):
    '''Runs padel for molecule.smi file'''
    if not os.path.isfile("results/" + chembl + "/files/descriptors_output.csv"):
        os.system("java -Xms1G -Xmx1G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt "
              "-standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ "
              "-file results/" + chembl + "/files/descriptors_output.csv")

def model_descriptors(chembl):
    '''Calculate model descriptors for a proteins drugs'''
    df3 = pd.read_csv('results/' + chembl + '/files/pIC50.csv')
    selection = ['canonical_smiles', 'molecule_chembl_id']
    df3_selection = df3[selection]
    df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)
    df3_selection.to_csv('results/' + chembl + '/files/molecule.smi', sep='\t', index=False, header=False)

    padel(chembl)
    print("PADEL COMPLETED")
    df3_X = pd.read_csv('results/' + chembl + '/files/descriptors_output.csv')
    df3_X = df3_X.drop(columns=['Name'])
    df3_Y = df3['pIC50']
    dataset3 = pd.concat([df3_X, df3_Y], axis=1)
    dataset3.to_csv('results/' + chembl + '/files/pIC50_pubchem_fp.csv', index=False)

def lazy_predictor(chembl, pp):

    df = pd.read_csv('results/' + chembl + '/files/pIC50_pubchem_fp.csv')
    X = df.drop('pIC50', axis=1)
    Y = df.pIC50
    selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = selection.fit_transform(X)

    # Perform data splitting using 80/20 ratio
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Defines and builds the lazyclassifier
    clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models_train, predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
    models_test, predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
    plots.model_stats(predictions_train, chembl, pp)

    # with open('results/' + chembl + '/models/lazy_pred_mod.pickle', 'wb') as handle:
    #     pickle.dump(models_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('results/' + chembl + '/models/lazy_pred_pred.pickle', 'wb') as handle:
    #     pickle.dump(predictions_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('results/' + chembl + '/models/lazy_pred_mod.pickle', 'rb') as handle:
    #     models_train = pickle.load(handle)
    # with open('results/' + chembl + '/models/lazy_pred_pred.pickle', 'rb') as handle:
    #     predictions_train = pickle.load(handle)

    # plots.model_stats(predictions_train, chembl, pp)
    return list(models_train.index)[0]




def hypertune_top_model(chembl, pp):
    '''Hypertune top model and save it'''
    print("Tuning HyperParameters")
    df = pd.read_csv('results/' + chembl + '/files/pIC50_pubchem_fp.csv')
    # df = df.dropna()
    X = df.drop('pIC50', axis=1)

    Y = df.pIC50

    selection = VarianceThreshold(threshold=(.8 * (1 - .8)))

    X = selection.fit_transform(X)

    selec = list(selection.get_support(indices=True))
    # print(list(selec))

    desc_list = pd.read_csv('results/' + chembl + '/files/pIC50_pubchem_fp.csv', usecols=selec)
    desc_list.to_csv('results/' + chembl + '/files/desc_list.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    np.random.seed(100)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)
    r2 = model.score(X_test, Y_test)
    Y_pred = model.predict(X_test)
    # plots.exp_vs_pred_ic50(Y_test, Y_pred, chembl, pp)
    print('results/' + chembl + '/models/ml_model.pickle')
    with open('results/' + chembl + '/models/ml_model.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model Tuned")




