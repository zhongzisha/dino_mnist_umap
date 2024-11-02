import os
import pickle
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns

postfix = '_official'
postfix = '_300'
postfix = '_300_final'
postfix = '_300_v2'
postfix = '_300_v2_final'
postfix = '_300_v1_final'

with open(f'val_feats{postfix}.pkl', 'rb') as fp:
    tmpdata = pickle.load(fp)
    val_feats0 = tmpdata['feats']
    val_labels0 = tmpdata['labels']
    del tmpdata


for selected_labels in [
    [0, 1, 3],
    [0, 3, 8, 5],
    [0, 8],
    [1, 9], 
    [6, 9, 0], 
    [1, 4, 6],
    [],
    [2, 3, 4]
]:
    name ='umap'
    save_dir = f'./umap{postfix}/'
    file_name='_'.join([str(i) for i in selected_labels])
    if os.path.exists(f"{save_dir}{file_name}{name}.png"):
        continue
    
    if len(selected_labels)>0:
        indexes = np.array([ind for ind, label in enumerate(val_labels0) if label in selected_labels])
        val_feats = val_feats0[indexes, :]
        val_labels = val_labels0[indexes]
    else:
        val_feats = val_feats0
        val_labels = val_labels0

    features = val_feats
    class_labels = val_labels
    os.makedirs(save_dir, exist_ok=True)
    n_neighbors=30
    min_dist=0.4
    spread=1.1
    epochs=100
    metric='euclidean'
    

    reducer = umap.UMAP(random_state=42, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        spread=spread,
        n_epochs=epochs,
        metric=metric, 
        n_components=2)
    umap_embedding = reducer.fit_transform(features)




    custom_palette = sns.color_palette("hls", len(set(class_labels)))

    def make_plot(embedding, labels, save_dir, file_name=file_name,name="Emb type", description="details"):
        sns_plot = sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels, s=14, palette=custom_palette, linewidth=0, alpha=0.9)
        plt.suptitle(f"{name}_{file_name}", fontsize=8)
        sns_plot.tick_params(labelbottom=False)
        sns_plot.tick_params(labelleft=False)
        sns_plot.tick_params(bottom=False)
        sns_plot.tick_params(left=False)
        sns_plot.set_title("CLS Token embedding of "+str(len(labels))+" cells with a dimensionality of "+str(features.shape[1])+" \n"+description, fontsize=6)
        sns.move_legend(sns_plot, "lower left", title='Classes', prop={'size': 5}, title_fontsize=6, markerscale=0.5)
        sns.set(rc={"figure.figsize":(14, 10)})
        sns.despine(bottom = True, left = True)
        sns_plot.figure.savefig(f"{save_dir}{file_name}{name}.png", dpi=325)
        # sns_plot.figure.savefig(f"{save_dir}pdf_format/{file_name}{name}.pdf")
        plt.close()

    make_plot(umap_embedding, class_labels, save_dir=save_dir, file_name=file_name, name="umap",description=f"n_neighbors:{n_neighbors}, min_dist={min_dist}, metric={metric}, spread={spread}, epochs={epochs}")















