from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.patches as patches

class Visualizer:
    def __init__(self):
        """初始化Visualizer类"""
        self.tsne = TSNE(n_components=2, random_state=42)
        
    def _get_spline_boundary(self, points, smoothness=0.2):
        """计算平滑的边界曲线点
        
        Args:
            points (numpy.ndarray): 边界点
            smoothness (float): 平滑程度，值越小曲线越平滑
            
        Returns:
            tuple: (x坐标数组, y坐标数组)
        """
        # 使用凸包算法找到边界点
        hull = ConvexHull(points)
        boundary_points = points[hull.vertices]
        
        # 闭合边界
        boundary_points = np.vstack((boundary_points, boundary_points[0]))
        
        # 在边界点之间插入更多的点以使曲线更平滑
        t = np.arange(len(boundary_points))
        t_smooth = np.linspace(0, len(boundary_points)-1, int(len(boundary_points)/smoothness))
        
        # 使用样条插值
        from scipy.interpolate import splprep, splev
        tck, u = splprep([boundary_points[:, 0], boundary_points[:, 1]], s=0, per=True)
        x_smooth, y_smooth = splev(np.linspace(0, 1, 1000), tck)
        
        return x_smooth, y_smooth
        
    def visualize(self, X, labels, save_path='cluster_result.png'):
        """使用t-SNE进行可视化
        
        Args:
            X (numpy.ndarray): 高维数据矩阵
            labels (numpy.ndarray): 聚类标签
            save_path (str): 保存图片的路径
        """
        # 使用t-SNE降维到2维
        X_2d = self.tsne.fit_transform(X)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建新图
        plt.figure(figsize=(12, 10))
        
        # 设置颜色映射
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        # 绘制散点图和边界
        for i in range(len(np.unique(labels))):
            mask = labels == i
            cluster_points = X_2d[mask]
            
            if len(cluster_points) < 4:  # 如果点太少，只画散点
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                          c=colors[i % len(colors)],
                          label=f'簇 {i+1}',
                          alpha=0.6)
                continue
                
            # 绘制该簇的所有点
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=colors[i % len(colors)],
                      label=f'簇 {i+1}',
                      alpha=0.6)
            
            # 计算该簇的中心点
            center = np.mean(cluster_points, axis=0)
            
            # 绘制中心点（使用★符号）
            plt.scatter(center[0], center[1],
                      c='black',
                      marker='*',
                      s=300,
                      label=f'簇 {i+1} 中心')
            
            # 计算并绘制平滑边界
            try:
                x_smooth, y_smooth = self._get_spline_boundary(cluster_points)
                plt.plot(x_smooth, y_smooth, '--', 
                        color=colors[i % len(colors)],
                        linewidth=2,
                        alpha=0.8)
                
                # 填充区域（透明）
                plt.fill(x_smooth, y_smooth,
                        color=colors[i % len(colors)],
                        alpha=0.1)
            except:
                print(f"无法为簇 {i+1} 生成平滑边界，点数可能太少或分布不合适")
        
        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加标题和轴标签
        plt.title('文本聚类结果可视化\n(★表示簇中心点，曲线表示簇的边界)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        
        # 调整布局以确保图例完全显示
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close() 