from phase2_gpu_figures import Phase2Figures, ROOT, NEURAL, LABELS, STYLE, plt, np, PdfPages

factual_only_val = np.array([[82.2, 12.8, 2.4, 1.5, 0.1, 0.4, 0.1, 0.5],
                             [14.8, 77.0, 5.5, 1.0, 0.0, 0.0, 0.8, 0.9],
                             [1.0, 26.6, 69.8, 0.0, 2.4, 0.2, 0.0, 0.0],
                             [9.5, 37.5, 0.0, 53.0, 0.0, 0.0, 0.0, 0.0],
                             [0.1, 38.9, 3.0, 0.0, 55.6, 2.4, 0.0, 0.0],
                             [8.5, 13.9, 2.9, 0.0, 36.4, 38.3, 0.0, 0.0],
                             [0.8, 66.1, 0.0, 0.0, 0.0, 0.0, 30.0, 3.1],
                             [10.6, 45.5, 0.0, 0.0, 0.0, 0.0, 13.7, 30.2]])
factual_only_tes = np.array([[85.5, 12.4, 1.0, 0.4, 0.1, 0.4, 0.1, 0.0],
                             [13.9, 80.1, 4.7, 0.3, 0.2, 0.0, 0.4, 0.4],
                             [5.2, 15.8, 77.3, 0.0, 1.7, 0.0, 0.0, 0.0],
                             [9.9, 46.0, 0.0, 44.0, 0.0, 0.0, 0.0, 0.0],
                             [0.2, 25.9, 32.4, 0.0, 40.9, 0.6, 0.0, 0.0],
                             [22.3, 4.6, 9.6, 0.0, 5.5, 58.0, 0.0, 0.0],
                             [0.0, 61.2, 0.0, 0.0, 0.0, 0.0, 35.0, 3.8],
                             [6.7, 44.5, 0.0, 0.0, 0.0, 0.0, 12.4, 36.4]])

full_analytic_val = np.array([[84.9, 10.8, 0.7, 0.8, 0.1, 0.7, 0.1, 1.9],
                              [17.9, 79.9, 0.8, 0.8, 0.1, 0.0, 0.0, 0.5],
                              [0.3, 0.8, 98.2, 0.0, 0.6, 0.0, 0.0, 0.0],
                              [12.7, 39.7, 0.0, 47.6, 0.0, 0.0, 0.0, 0.0],
                              [0.3, 43.3, 1.8, 0.0, 53.1, 1.6, 0.0, 0.0],
                              [8.5, 23.5, 3.7, 0.0, 39.2, 25.1, 0.0, 0.0],
                              [1.0, 65.6, 0.0, 0.0, 0.0, 0.0, 31.5, 1.9],
                              [2.2, 3.3, 0.0, 0.0, 0.0, 0.0, 24.5, 70.0]])
full_analytic_tes = np.array([[87.5, 9.9, 0.5, 0.5, 0.0, 0.5, 0.1, 1.0],
                              [17.7, 79.9, 0.9, 0.4, 0.1, 0.0, 0.2, 0.9],
                              [0.0, 0.0, 99.8, 0.0, 0.2, 0.0, 0.0, 0.0],
                              [12.9, 46.5, 0.0, 40.6, 0.0, 0.0, 0.0, 0.0],
                              [0.2, 28.8, 49.3, 0.0, 21.1, 0.5, 0.0, 0.0],
                              [21.4, 9.6, 12.2, 0.0, 2.0, 54.8, 0.0, 0.0],
                              [0.0, 76.4, 0.0, 0.0, 0.0, 0.0, 21.0, 2.6],
                              [8.0, 31.8, 0.0, 0.0, 0.0, 0.0, 14.0, 46.2]])

full_val = np.array([[91.0, 6.3, 0.8, 0.0, 0.1, 0.1, 0.2, 1.5],
                     [7.5, 89.3, 1.2, 1.5, 0.3, 0.0, 0.0, 0.4],
                     [0.0, 0.5, 99.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [6.1, 27.2, 0.0, 66.7, 0.0, 0.0, 0.0, 0.0],
                     [0.1, 35.8, 1.2, 0.0, 62.0, 0.9, 0.0, 0.0],
                     [8.0, 9.1, 4.5, 0.0, 16.3, 62.1, 0.0, 0.0],
                     [0.0, 33.8, 0.0, 0.0, 0.0, 0.0, 60.8, 5.4],
                     [0.4, 3.7, 0.0, 0.0, 0.0, 0.0, 3.3, 92.7]])
full_tes = np.array([[91.3, 6.7, 0.7, 0.0, 0.0, 0.1, 0.1, 1.0],
                     [7.0, 91.0, 0.7, 0.5, 0.7, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 99.8, 0.0, 0.2, 0.0, 0.0, 0.0],
                     [4.3, 20.5, 0.0, 75.2, 0.0, 0.0, 0.0, 0.0],
                     [0.0, 19.4, 14.4, 0.0, 66.0, 0.2, 0.0, 0.0],
                     [9.2, 7.4, 8.7, 0.0, 2.3, 72.4, 0.0, 0.0],
                     [0.0, 30.3, 0.0, 0.0, 0.0, 0.0, 69.7, 0.0],
                     [0.0, 15.7, 0.0, 0.0, 0.0, 0.0, 8.4, 75.9]])

matrix_val = {'factual_only': factual_only_val, 'full_without_analytic': full_analytic_val, 'full': full_val}
matrix_tes = {'factual_only': factual_only_tes, 'full_without_analytic': full_analytic_tes, 'full': full_tes}


class Confusion1x3Figures(Phase2Figures):
    def __init__(self):
        super().__init__(output=ROOT / 'figures/confusion_1x3_v3')

    def confusion(self, split):
        im = None
        if split == "test":
            matrix_dic = matrix_tes
        else:
            matrix_dic = matrix_val
        fig, axes = plt.subplots(1, 3, figsize=(9.48, 3.2))
        fig.subplots_adjust(left=0.065, right=0.88, bottom=0.15, top=0.90, wspace=0.20)
        # fig.suptitle(f'{split.capitalize()} confusion matrices', y=0.985, fontsize=9)
        # fig.text(0.47, 0.905, 'Rows: reference class; columns: predicted class', ha='center', fontsize=7)
        for index, (key, ax) in enumerate(zip(NEURAL, axes)):
            matrix = matrix_dic[key]
            im = ax.imshow(matrix, cmap='cividis', vmin=0, vmax=100, interpolation='nearest')
            ax.set_title(f'({chr(97 + index)}) {LABELS[key]}', loc='left', fontsize=12, pad=8)
            ax.set_xticks(range(8), [f'C{i}' for i in range(8)])
            ax.set_yticks(range(8), [f'C{i}' for i in range(8)])
            ax.tick_params(length=0, pad=3, labelsize=9)
            ax.set_xlabel('Predicted class', fontsize=12)
            ax.set_ylabel('Reference class', fontsize=12)
            for i in range(8):
                for j in range(8):
                    value = matrix[i, j]
                    ax.text(j, i, '0' if value == 0 else f'{value:.1f}', ha='center', va='center', fontsize=8,
                            color='black' if value > 45 else 'white')
        cax = fig.add_axes([0.915, 0.17, 0.015, 0.7])
        bar = fig.colorbar(im, cax=cax)
        bar.set_label('Row percentage (%)', fontsize=11)
        bar.ax.tick_params(labelsize=9)
        caption = (
            f'Confusion matrices on the same 2,500 {split} samples. '
            'Rows are dominant reference pseudo-classes C0-C7; columns are predictions. '
            'Each panel is the elementwise mean of five row-normalized seed matrices, '
            'not an ensemble. Checkpoints were selected by minimum validation NLL. '
            'The three neural models share the same 0-100% scale. '
            'Type prior is omitted from this figure; all saved evaluation data are unchanged.'
        )
        plt.show()
        # self.save(fig, f'confusion_{split}_three_models_1x3', f'{split}：三个神经模型混淆矩阵（一行三列）', caption)

    def run(self):
        self.read_training()
        self.read_evaluation()
        plt.rcParams.update(STYLE)
        self.confusion('test')
        # self.confusion('validation')


if __name__ == '__main__':
    plt.switch_backend('TkAgg')
    Confusion1x3Figures().run()
