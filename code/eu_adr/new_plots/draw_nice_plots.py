import numpy as np
from matplotlib import pyplot as plt
import pickle


def draw_learning_comparison(splits, r_score, u_score, d_score, samples_per_split, scoring, min, max, folder,
                             cross_valid, original):
    """
    Plot the different learning methods on same graph
    """
    # if curve being plotted is for original data only
    # the values used here are dependent on the size of the data sets
    if original:
        orig_size = 0
        num_records = 1201
    else:
        num_records = 486
        orig_size = 1201

    # create ticks for x axis
    if cross_valid:
        ticks = np.linspace(orig_size + samples_per_split, orig_size + (splits*samples_per_split), splits)
    else:
        ticks = np.linspace(orig_size + samples_per_split, orig_size + ((splits-1)*samples_per_split), splits-1)
        # final training set included rounding errors so use correct value instead of linspace above
        total_num_training = num_records + orig_size - (num_records/5)
        ticks = np.append(ticks, total_num_training)

    print samples_per_split
    print ticks

    # set up the figure
    plt.figure()
    plt.grid()
    plt.xlabel('Training Instances')

    if original:
        plt.xlim([0, 1000])
    else:
        plt.xlim([1200, 1600])

    plt.ylabel(scoring)

    # set the axis limits
    buffer_space = (max - min)/10.0
    plt.ylim([min - buffer_space, max + buffer_space])
    plt.title('%s comparison using %s batches' % (scoring, splits))

    plt.plot(ticks, r_score, label='Random Sampling')
    plt.plot(ticks, u_score, label='Uncertainty Sampling')
    plt.plot(ticks, d_score, label='Density Sampling')

    plt.legend(loc='best')
    # may want to force legend to bottom right in some cases
    #plt.legend(loc=4)

    f_name = '%s/%s_%s_splits.eps' % (folder, scoring, splits)
    # minimise the borders so they fit more nicely in report
    # eps format for latex
    plt.savefig(f_name, format='eps', bbox_inches='tight')
    plt.close()


def do_dem(folder, cross_valid=False, original=True):
    """
    Draw plots for the passed in pickles
    """
    scores_5 = pickle.load(open(folder + '/splits5.p', 'rb'))
    scores_10 = pickle.load(open(folder + '/splits10.p', 'rb'))
    scores_20 = pickle.load(open(folder + '/splits20.p', 'rb'))
    scores_40 = pickle.load(open(folder + '/splits40.p', 'rb'))

    # concatenate all scores to calculate axis limits via min, max
    a = np.concatenate((np.concatenate(scores_5[0][1:]), np.concatenate(scores_10[0][1:]),
                        np.concatenate(scores_20[0][1:]), np.concatenate(scores_40[0][1:])))

    p = np.concatenate((np.concatenate(scores_5[1][1:]), np.concatenate(scores_10[1][1:]),
                        np.concatenate(scores_20[1][1:]), np.concatenate(scores_40[1][1:])))

    r = np.concatenate((np.concatenate(scores_5[2][1:]), np.concatenate(scores_10[2][1:]),
                        np.concatenate(scores_20[2][1:]), np.concatenate(scores_40[2][1:])))

    f = np.concatenate((np.concatenate(scores_5[3][1:]), np.concatenate(scores_10[3][1:]),
                        np.concatenate(scores_20[3][1:]), np.concatenate(scores_40[3][1:])))

    mins = [a.min(), p.min(), r.min(), f.min()]
    maxs = [a.max(), p.max(), r.max(), f.max()]

    for i in xrange(4):
        if original:
            samples_per_split = (4 * 1201)/(5 * 5)
        else:
            samples_per_split = (4 * 486)/(5 * 5)

        draw_learning_comparison(5, scores_5[i][1], scores_5[i][2], scores_5[i][3], samples_per_split, scores_5[i][0],
                                 mins[i], maxs[i], folder, cross_valid, original)

        samples_per_split /= 2
        draw_learning_comparison(10, scores_10[i][1], scores_10[i][2], scores_10[i][3], samples_per_split,
                                 scores_10[i][0], mins[i], maxs[i], folder, cross_valid, original)

        samples_per_split /= 2
        draw_learning_comparison(20, scores_20[i][1], scores_20[i][2], scores_20[i][3], samples_per_split,
                                 scores_20[i][0], mins[i], maxs[i], folder, cross_valid, original)

        samples_per_split /= 2
        draw_learning_comparison(40, scores_40[i][1], scores_40[i][2], scores_40[i][3], samples_per_split,
                                 scores_40[i][0], mins[i], maxs[i], folder, cross_valid, original)

if __name__ == '__main__':
    #do_dem('1_orig_bag_of_words')
    #do_dem('2_orig_bag_of_words_cross_valid', True)
    #do_dem('3_orig_features_rbf')
    #do_dem('4_orig_features_linear')
    #do_dem('5_new_data_bag_of_words', original=False)
    #do_dem('6_new_data_features_linear', original=False)
    #do_dem('7_orig_active_words')
    #do_dem('8_new_data_active_words', original=False)
    do_dem('9_new_data_features_rbf', original=False)
