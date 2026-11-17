"""复现用户提供的六模型混淆矩阵：直接运行显示，自行保存图片。

数值为截图中的百分数，行/列均按 C0–C7 排列；保留舍入值，不重新归一化。
"""
import matplotlib.pyplot as plt
import numpy as np


rf_tes = np.array([
    [89.1, 9.7, 0.3, 0, 0, 0, 0, 0.9],
    [9.6, 89.4, 0.6, 0.4, 0, 0, 0, 0],
    [0, 0, 100, 0, 0, 0, 0, 0],
    [8.8, 20.5, 0, 70.7, 0, 0, 0, 0],
    [0, 26.7, 52.3, 0, 20.9, 0.1, 0, 0],
    [9.1, 10.8, 17.0, 0, 2.4, 60.6, 0, 0],
    [0, 52.3, 0, 0, 0, 0, 47.7, 0],
    [0, 28.2, 0, 0, 0, 0, 4.2, 67.6],
])

factual_tes = np.array([
    [85.5, 12.4, 1.0, 0.4, 0.1, 0.4, 0.1, 0],
    [13.9, 80.1, 4.7, 0.3, 0.2, 0, 0.4, 0.4],
    [5.2, 15.8, 77.3, 0, 1.7, 0, 0, 0],
    [9.9, 46.0, 0, 44.0, 0, 0, 0, 0],
    [0.2, 25.9, 32.4, 0, 40.9, 0.6, 0, 0],
    [22.3, 4.6, 9.6, 0, 5.5, 58.0, 0, 0],
    [0, 61.2, 0, 0, 0, 0, 35.0, 3.8],
    [6.7, 44.5, 0, 0, 0, 0, 12.4, 36.4],
])

bilstm_tes = np.array([
    [86.3, 12.0, 1.2, 0.3, 0, 0.1, 0, 0],
    [11.4, 82.6, 4.4, 0.4, 0.5, 0, 0, 0.7],
    [1.0, 8.7, 90.2, 0, 0.2, 0, 0, 0],
    [10.4, 42.0, 0, 47.6, 0, 0, 0, 0],
    [0.2, 22.7, 49.0, 0, 27.7, 0.3, 0, 0],
    [19.7, 3.8, 16.5, 0, 6.4, 53.6, 0, 0],
    [0, 50.9, 0, 0, 0, 0, 47.5, 1.6],
    [6.4, 63.7, 0, 0, 0, 0, 28.2, 1.7],
])

inception_tes = np.array([
    [84.0, 13.6, 1.0, 1.0, 0, 0, 0.1, 0.2],
    [11.4, 81.7, 5.3, 0.4, 0.1, 0, 0.3, 0.9],
    [1.2, 7.0, 91.7, 0, 0.2, 0, 0, 0],
    [8.7, 37.1, 0, 54.3, 0, 0, 0, 0],
    [0.2, 25.2, 49.6, 0, 24.7, 0.3, 0, 0],
    [22.6, 8.7, 11.6, 0, 2.9, 54.2, 0, 0],
    [0.2, 60.2, 0, 0, 0, 0, 35.6, 4.0],
    [7.1, 52.9, 0, 0, 0, 0, 20.9, 19.1],
])

transformer_tes = np.array([
    [85.0, 11.5, 1.4, 1.1, 0, 0.2, 0, 0.6],
    [13.5, 75.9, 6.8, 0.6, 0.8, 0, 0.3, 2.2],
    [1.5, 4.3, 93.5, 0, 0.7, 0, 0, 0],
    [8.5, 18.0, 0, 73.5, 0, 0, 0, 0],
    [0.2, 20.2, 42.6, 0, 36.4, 0.5, 0, 0],
    [16.5, 10.1, 12.5, 0, 1.4, 59.4, 0, 0],
    [0, 55.0, 0, 0, 0, 0, 32.7, 12.3],
    [6.5, 56.8, 0, 0, 0, 0, 17.4, 19.3],
])

full_tes = np.array([
    [91.3, 6.7, 0.7, 0, 0, 0.1, 0.1, 1.0],
    [7.0, 91.0, 0.7, 0.5, 0.7, 0, 0, 0],
    [0, 0, 99.8, 0, 0.2, 0, 0, 0],
    [4.3, 20.5, 0, 75.2, 0, 0, 0, 0],
    [0, 19.4, 14.4, 0, 66.0, 0.2, 0, 0],
    [9.2, 7.4, 8.7, 0, 2.3, 72.4, 0, 0],
    [0, 30.3, 0, 0, 0, 0, 69.7, 0],
    [0, 15.7, 0, 0, 0, 0, 8.4, 75.9],
])

MODELS = (
    ('analytic_random_forest', 'Analytic RF'),
    ('factual_only', 'Factual TCN'),
    ('bilstm_attention', 'BiLSTM + Attention'),
    ('inception_time', 'InceptionTime'),
    ('lightweight_transformer', 'Transformer'),
    ('full', 'Full'),
)


class Phase3BaselineConfusion:
    def read_data(self):
        self.matrices = dict(zip(
            (key for key, _ in MODELS),
            (rf_tes, factual_tes, bilstm_tes, inception_tes, transformer_tes, full_tes),
        ))
        for key, matrix in self.matrices.items():
            # 八个单元格各保留一位小数，行和允许舍入偏差，不改动图中数值。
            if (matrix.shape != (8, 8) or not np.isfinite(matrix).all()
                    or np.any(matrix < 0) or np.any(matrix > 100)
                    or not np.allclose(matrix.sum(axis=1), 100, atol=0.4, rtol=0)):
                raise ValueError(f'Invalid row percentages: {key}.')

    def plot(self):
        plt.rcParams.update({
            'font.family': 'Times New Roman', 'font.size': 9, 'pdf.fonttype': 42,
            'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': 0.6,
        })
        fig, axes = plt.subplots(2, 3, figsize=(8.48, 5.25))
        fig.subplots_adjust(left=0.06, right=0.88, bottom=0.1, top=0.94,
                            wspace=0.1, hspace=0.21)
        for ax, (key, label) in zip(axes.flat, MODELS):
            matrix = self.matrices[key]
            mesh = ax.pcolormesh(np.arange(9) - 0.5, np.arange(9) - 0.5,
                                 matrix, cmap='cividis', vmin=0, vmax=100,
                                 shading='flat', rasterized=False)
            ax.set(xlim=(-0.5, 7.5), ylim=(7.5, -0.5), aspect='equal')
            ax.set_title(label, fontsize=12, pad=7)
            ax.set_xticks(range(8), [f'C{i}' for i in range(8)])
            ax.set_yticks(range(8), [f'C{i}' for i in range(8)])
            ax.tick_params(length=0, labelsize=10, pad=3)
            for i in range(8):
                for j in range(8):
                    value = matrix[i, j]
                    text = '0' if value == 0 else ('100' if value == 100 else f'{value:.1f}')
                    ax.text(j, i, text, ha='center', va='center', fontsize=7.2,
                            color='black' if value > 45 else 'white')
        fig.supxlabel('Predicted class', fontsize=12, y=0.02)
        fig.supylabel('Reference class', fontsize=12, x=0.009)
        colorbar = fig.colorbar(mesh, cax=fig.add_axes([0.905, 0.195, 0.015, 0.64]))
        colorbar.solids.set_rasterized(False)
        colorbar.set_label('Row percentage (%)', fontsize=12, labelpad=6)
        colorbar.set_ticks([0, 20, 40, 60, 80, 100])
        colorbar.ax.tick_params(labelsize=10, length=2)
        return fig

    def run(self, show=True):
        self.read_data()
        fig = self.plot()
        if show:
            plt.show()
        return fig


if __name__ == '__main__':
    Phase3BaselineConfusion().run()
