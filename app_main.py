import app_lib as lib
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def build_model(chembl):
    # chembl = "CHEMBL220"

    if lib.does_exist(chembl) == False:
        pp = PdfPages('results/' + chembl + '/figures/multipage.pdf')
        df = lib.target_search(chembl)
        df2 = lib.preprocess(df, chembl)
        df_lipinski = lib.lipinski(df2.canonical_smiles)
        df_combined = pd.concat([df2,df_lipinski], axis=1)
        df_2class = lib.eda(df_combined, chembl)
        # df_2class = pd.read_csv('results/' + parsed.chembl + '/files/pIC50.csv')
        lib.visual_eda(df_2class, chembl, pp)

        lib.model_descriptors(chembl)
        best_model = lib.lazy_predictor(chembl, pp)
        print(best_model)
        lib.hypertune_top_model(chembl, pp)
        pp.close()


# build_model("CHEMBL220")