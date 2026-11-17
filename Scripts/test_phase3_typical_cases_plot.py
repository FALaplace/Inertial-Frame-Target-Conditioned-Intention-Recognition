"""使用冻结个例检查模式切换；直接运行，不显示或导出图片。"""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from phase3_typical_cases_plot import Phase3TypicalCasesPlot


class TypicalCaseModesTest(unittest.TestCase):
    def test_case_modes(self):
        for case, index, fields, banned in (
            ('c4_c2', 17985, ('distance_km', 'range_rate_km_s'), (6, 7)),
            ('c7_c1', 12976, ('elevation_deg', 'access_flag'), (2, 3, 4)),
        ):
            with self.subTest(case=case):
                plotter = Phase3TypicalCasesPlot()
                fig = plotter.run(show=False, cases=(case,))
                self.addCleanup(plt.close, fig)
                self.assertEqual(set(plotter.physical), {case})
                self.assertEqual(plotter.physical[case]['sample_index'], index)
                self.assertEqual([ax.name for ax in fig.axes[:2]], ['3d', '3d'])
                trajectory = plotter.trajectories[case]
                for ax, key in zip(fig.axes[:2], ('actor_r', 'actor_lvlh')):
                    np.testing.assert_array_equal(np.asarray(ax.lines[0].get_data_3d()).T, trajectory[key])
                np.testing.assert_allclose(np.linalg.norm(trajectory['actor_lvlh'], axis=1),
                                           plotter.physical[case]['factual']['distance_km'], atol=1e-6)
                offset = 2
                for ax, field in zip(fig.axes[offset:offset + 2], fields):
                    np.testing.assert_array_equal(ax.lines[0].get_ydata(), plotter.physical[case]['factual'][field])
                heatmap = fig.axes[offset + 2].collections[0].get_array().reshape(7, 8)
                np.testing.assert_array_equal(heatmap.mask[0], [i in banned for i in range(8)])
                np.testing.assert_array_equal(heatmap.data, plotter.probabilities[case])
                if case == 'c7_c1':
                    self.assertEqual(fig.axes[3].lines[0].get_drawstyle(), 'steps-post')
                    target = trajectory['counterpart_r']
                    up = target / np.linalg.norm(target, axis=1, keepdims=True)
                    east = np.column_stack([-target[:, 1], target[:, 0], np.zeros(len(target))])
                    east /= np.linalg.norm(east, axis=1, keepdims=True)
                    north = np.cross(up, east)
                    displacement = trajectory['actor_r'] - target
                    expected = np.column_stack([np.sum(displacement * axis, axis=1) for axis in (up, east, north)])
                    np.testing.assert_allclose(trajectory['actor_lvlh'], expected, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
