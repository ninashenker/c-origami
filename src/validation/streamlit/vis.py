import os
import sys
sys.path.append('../')
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.measure import block_reduce

mm = 1/25.4  # centimeters in inches

# Pre-processing
def ctcf_proc(ctcf, model_st):
    if model_st in ['tomato', 'eggplant', 'pear', 'pineapple']:
        return np.exp2(ctcf)
    elif model_st in ['peach', 'grape', 'durian', 'mango', 'mango_seq']:
        return ctcf
    else:
        raise Exception('New model, please add category')

def atac_proc(atac, model_st):
    return atac

# Plotting
def plot_mat(handle, image, offset = 0, img_type = 'truth', cell_type = 'None', loci = 'None', size = 40):
    from matplotlib.colors import LinearSegmentedColormap
    mm = 1/25.4  # centimeters in inches
    offset /=  1000000
    color_map = LinearSegmentedColormap.from_list("bright_red",
                                                [(1,1,1),(1,0,0)])
    #fig, ax = plt.subplots(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(size*mm, size*mm))
    ax.imshow(image, cmap = color_map, vmin = 0, vmax = 5)
    ax.set_xticks([0,100,200])
    ax.set_yticks([0,100,200])

    if 'pred' in img_type and loci[0] == 'chr15' and cell_type != 'IMR-90':
        #ax.set_axis_off()
        norm = plt.Normalize(0, 5)
        sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
        cbar.set_label('log(intensity)')
    hic_chart = handle.pyplot(fig)
    save_dir = './save'
    if not os.path.exists('./save'): os.makedirs('./save')
    plt.savefig(f'./save/{cell_type}_{"_".join(loci)}_{img_type}.pdf', bbox_inches = 'tight')
    plt.clf()

def plot_diff(handle, image, offset = 0, vmax = 3, num = 0.141343123, img_type = 'truth', cell_type = 'None', loci = 'None', size = 40):
    offset /=  1000000
    '''
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0, 5]})
    ax[1].imshow(image, cmap = 'RdBu_r', vmin = - vmax, vmax = vmax)
    ax[1].set_axis_off()

    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlim([0 + offset, 2.097152 + offset])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].figure.set_size_inches(5, 7)
    '''
    #fig, ax = plt.subplots(figsize=(5, 5))
    fig, ax = plt.subplots(figsize=(size*mm, size*mm))
    ax.imshow(image, cmap = 'RdBu_r', vmin = - vmax, vmax = vmax)
    ax.set_xticks([0,100,200])
    ax.set_yticks([0,100,200])
    if 'pred' in img_type and loci[0] == 'chr15':
        #ax.set_axis_off()
        norm = plt.Normalize(0, 5)
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
        cbar.set_label('Difference')
    hic_chart = handle.pyplot(fig)
    save_dir = './save'
    if not os.path.exists('./save'): os.makedirs('./save')
    plt.savefig(f'./save/{cell_type}_{"_".join(loci)}_{img_type}.pdf', bbox_inches = 'tight')
    plt.clf()

def plot_diff_auto_range(handle, image, num = 0.141343123):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0, 5]})
    ax[1].imshow(image, cmap = 'RdBu_r')
    ax[1].set_axis_off()

    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlim([0, 209.7152])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].figure.set_size_inches(5, 7)
    hic_chart = handle.pyplot(fig)

def save_diff(name, image):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0, 5]})
    ax[1].imshow(image, cmap = 'RdBu_r', vmin = -3, vmax = 3)
    ax[1].set_axis_off()

    ax[0].get_yaxis().set_visible(False)
    ax[0].set_xlim([0, 209.7152])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].figure.set_size_inches(5, 7)
    if not os.path.exists('imgs'): os.mkdir('imgs')
    plt.savefig(f'imgs/{name}.png')
    plt.clf()

def plot_mat_epi(handle, image, ctcf, atac):
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 0, 0.5, 0.5]})
    color_map = LinearSegmentedColormap.from_list("bright_red",
                                                [(1,1,1),(1,0,0)])
    ax[0].imshow(image, cmap = color_map, vmin = 0, vmax = 5)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])

    ax[2].plot(ctcf)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].margins(x=0)
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    ax[3].plot(atac, color='tab:orange')
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].margins(x=0)
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)

    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlim([0, 209.7152])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].figure.set_size_inches(5, 7)
    handle.pyplot(fig)

def plot_diff_epi(handle, image, ctcf, atac, vmax = 3, num = 0.141343123):
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 0, 0.5, 0.5]})
    ax[0].imshow(image, cmap = 'RdBu_r', vmin = -vmax, vmax = vmax)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])

    ax[2].plot(ctcf)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['bottom'].set_visible(False)
    ax[2].margins(x=0)
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)

    ax[3].plot(atac, color='tab:orange')
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['bottom'].set_visible(False)
    ax[3].margins(x=0)
    ax[3].get_xaxis().set_visible(False)
    ax[3].get_yaxis().set_visible(False)

    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlim([0, 209.7152])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].figure.set_size_inches(5, 7)
    handle.pyplot(fig)

def legacy_plot_diff_epi(handle, image, ctcf, atac, num = 0.141343123):
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [5, 0, 0.5, 0.5]})
    ax[0].imshow(image, cmap = 'RdBu_r', vmin = -3, vmax = 3)
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])

    ax[2].plot(ctcf)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].margins(x=0)
    ax[2].get_xaxis().set_visible(False)

    ax[3].plot(atac, color='tab:orange')
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].margins(x=0)
    ax[3].get_xaxis().set_visible(False)

    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlim([0, 209.7152])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].figure.set_size_inches(5, 7)
    handle.pyplot(fig)

def fill_plot(handle, data, img_type, cell_type, loci, assay, maxpool = True, size = 40, is_diff = False):
    from skimage.measure import block_reduce
    fig, ax = plt.subplots(figsize=(size*mm, size*mm*0.3))
    if maxpool:
        data = block_reduce(data, (4196,), np.max)
    ax.fill_between(range(len(data)), data, color = 'black', linewidth = 0.3)
    ax.margins(x=0)
    ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if assay == 'ctcf':
        if is_diff:
            plt.ylim(-2500, 2500)
        else:
            plt.ylim(0, 2500)
    else:
        if is_diff:
            plt.ylim(-7000, 7000)
        else:
            plt.ylim(0, 7000)
    handle.pyplot(fig)
    save_dir = './save'
    if not os.path.exists('./save'): os.makedirs('./save')
    plt.savefig(f'./save/{cell_type}_{"_".join(loci)}_{img_type}_{assay}.pdf', bbox_inches = 'tight')
    plt.clf()

def plot_epi(handle, ctcf, atac, img_type = 'truth', cell_type = 'None', loci = 'None'):

    fill_plot(handle, ctcf, img_type, cell_type, loci, 'ctcf')
    fill_plot(handle, atac, img_type, cell_type, loci, 'atac')
    #F0746E
    #FCDE9C
    #7C1D6F
    #F0746E
    #F0746E
    #045275
    #089099


def plot_epi_diff(handle, ctcf1, ctcf2, atac1, atac2, img_type = 'truth', cell_type = 'None', loci = 'None'):
    from skimage.measure import block_reduce
    ctcf = ctcf2 - ctcf1
    atac = atac2 - atac1
    max_func = lambda x, axis : x[np.arange(x.shape[0]), np.argmax(np.abs(x), axis = list(axis)[0])]
    #ctcf = block_reduce(ctcf2, (4196,), max_func) - block_reduce(ctcf1, (4196,), max_func)
    #atac = block_reduce(atac2, (4196,), max_func) - block_reduce(atac1, (4196,), max_func)
    ctcf = block_reduce(ctcf, (4196,), max_func)
    atac = block_reduce(atac, (4196,), max_func)

    fill_plot(handle, ctcf, img_type, cell_type, loci, 'ctcf', False, is_diff = True)
    fill_plot(handle, atac, img_type, cell_type, loci, 'atac', False, is_diff = True)

def get_insulation(matrix):
    from insulation_score import chr_score
    truth_scores = np.array(chr_score(matrix))
    truth_scores = (truth_scores - np.nanmean(truth_scores)) / np.nanstd(truth_scores)
    return truth_scores

def vis_insulation(handle, scores, img_type = 'truth', cell_type = 'None', loci = 'None', size = 40, dot_line = False, vmin = None, vmax = None):

    n = 0
    m = 256
    xs = np.arange(n, m)
    fig, ax = plt.subplots(figsize=(size*mm, size*mm*0.3))
    ax.margins(x=0)
    if dot_line:
        ax.plot(xs, scores[n : m], linestyle = 'dashdot', color = 'black')
    else:
        ax.plot(xs, scores[n : m], color = 'black')
    if not vmin is None:
        plt.ylim(vmin, vmax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handle.pyplot(fig)
    save_dir = './save'
    if not os.path.exists('./save'): os.makedirs('./save')
    plt.savefig(f'./save/{cell_type}_{"_".join(loci)}_{img_type}_insulation.pdf', bbox_inches = 'tight')

def vis_insulation_diff(handle, scores_1, scores_2, name1, name2, img_type = 'truth', cell_type = 'None', loci = 'None', size = 40):

    n = 0
    m = 256
    xs = np.arange(n, m)
    fig, ax = plt.subplots(figsize=(size*mm, 32*mm))
    ax.margins(x=0)
    ax.plot(xs, scores_1[n : m], label = name1, color = 'black')
    ax.plot(xs, scores_2[n : m], label = name2, linestyle = 'dashdot', color = 'black')
    from scipy.stats import pearsonr
    pearson_corr = pearsonr(scores_1[10:-10], scores_2[10:-10])[0]
    textstr = '$\it{r}$' + f' = {pearson_corr:.2f}'
    ax.text(0.7, 1.15, textstr, transform=ax.transAxes, verticalalignment='top')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-4, 4)

    ax.set_xticks([100,200])

    if img_type == 'legend':
        plt.legend(bbox_to_anchor=(1.1, 1.1, 1.1, 1.1), frameon=False)

    handle.pyplot(fig)
    save_dir = './save'
    if not os.path.exists('./save'): os.makedirs('./save')
    plt.savefig(f'./save/{cell_type}_{"_".join(loci)}_{img_type}_insulation.pdf', bbox_inches = 'tight')

def plot_insulation(handle, matrix, img_type = 'truth', cell_type = 'None', loci = 'None', size = 40, dot_line = False): 

    from insulation_score import chr_score

    truth_scores = np.array(chr_score(matrix))
    truth_scores = (truth_scores - np.nanmean(truth_scores)) / np.nanstd(truth_scores)
    n = 0
    m = 256
    xs = np.arange(n, m)
    fig, ax = plt.subplots(figsize=(size*mm, size*mm*0.3))
    ax.margins(x=0)
    if dot_line:
        ax.plot(xs, truth_scores[n : m], linestyle = 'dashdot', color = 'black')
    else:
        ax.plot(xs, truth_scores[n : m], label = 'New', color = 'black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handle.pyplot(fig)
    save_dir = './save'
    if not os.path.exists('./save'): os.makedirs('./save')
    plt.savefig(f'./save/{cell_type}_{"_".join(loci)}_{img_type}_insulation.pdf', bbox_inches = 'tight')

def plot_insulation_diff(handle, matrix_1, matrix_2, name1, name2, img_type = 'truth', cell_type = 'None', loci = 'None', size = 40):

    from insulation_score import chr_score

    scores_1 = np.array(chr_score(matrix_1))
    scores_1 = (scores_1 - np.nanmean(scores_1)) / np.nanstd(scores_1)
    scores_2 = np.array(chr_score(matrix_2))
    scores_2 = (scores_2 - np.nanmean(scores_2)) / np.nanstd(scores_2)

    vis_insulation_diff(handle, scores_1, scores_2, name1, name2, img_type = img_type, cell_type = cell_type, loci = loci, size = size)
