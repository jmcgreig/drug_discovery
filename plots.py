import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

def make_plots(freq, scatter, box, var, df_2class, chembl, pp):
    # Frequency Plot
    if freq == 1:
        plt.figure(figsize=(5.5, 5.5))

        sns.countplot(x='bioactivity_class', data=df_2class, edgecolor='black')

        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')
        # pp.savefig('results/' + chembl + '/figures/bioactivity_class.pdf')
        pp.savefig(bbox_inches='tight')

    # Scatter Plot
    if scatter == 1:
        plt.figure(figsize=(5.5, 5.5))

        sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='pIC50', edgecolor='black',
                        alpha=0.7)

        plt.xlabel('MW', fontsize=14, fontweight='bold')
        plt.ylabel('LogP', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        # pp.savefig('results/' + chembl + '/figures/MW_vs_LogP.pdf')
        pp.savefig(bbox_inches='tight')

    if box == 1:
        # Box Plots
        plt.figure(figsize=(5.5, 5.5))

        sns.boxplot(x='bioactivity_class', y=var, data=df_2class)

        plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
        plt.ylabel('%s value' % var, fontsize=14, fontweight='bold')
        # pp.savefig('results/' + chembl + '/figures/ic50.pdf')
        pp.savefig(bbox_inches='tight')


def model_stats(predictions_train, chembl, pp):
    params = {"R-Squared": 1, "RMSE" : 10, "Time Taken": 10}

    for k, v in params.items():
        plt.figure(figsize=(5.5, 5.5))
        sns.set_theme(style="whitegrid")

        ax = sns.barplot(y=predictions_train.index, x=k, data=predictions_train)
        ax.set(xlim=(0, v))
        # pp.savefig('results/' + chembl + '/figures/model_performance.pdf')
        pp.savefig(bbox_inches='tight')

def exp_vs_pred_ic50(Y_test, Y_pred, chembl, pp):
    plt.clf()

    sns.set(color_codes=True)
    sns.set_style("white")
    ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha': 0.4})
    ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
    ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.figure.set_size_inches(5.5, 5.5)
    # pp.savefig('results/' + chembl + '/figures/exp_vs_pred_ic50.pdf')
    pp.savefig(bbox_inches='tight')
