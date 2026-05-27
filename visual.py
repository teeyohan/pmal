import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score


def merge_by_species(all_data_by_task):

    # 合并任务(按物种)
    all_data = defaultdict(list)
    human_tensors = []

    # 处理all_data字典
    for key, tensor in all_data_by_task.items():
        # 提取A部分（第一个下划线前的内容）
        prefix = key.split('_')[0]
        if prefix == 'mammal (species unspecified)':
            prefix = 'mammal'
        if prefix == 'bird-wild':
            prefix = 'bird'
        if prefix in ['man', 'women', 'human']:
            human_tensors.append(tensor)
            continue
        all_data[prefix].append(tensor)

    if human_tensors:
        all_data['human'].append(torch.cat(human_tensors, dim=0))

    # 将列表中的Tensor拼接起来
    for prefix in all_data:
        if len(all_data[prefix]) > 1:
            # 如果有多个Tensor，在dim=0维度上拼接
            all_data[prefix] = torch.cat(all_data[prefix], dim=0)
        else:
            # 如果只有一个Tensor，直接取出
            all_data[prefix] = all_data[prefix][0]

    return all_data


def plot_predictions_vs_actual_scatter(all_preds, all_labels):
    """
    绘制预测值vs真实值散点图
    
    Parameters:
    -----------
    all_preds : dict
        任务名称到预测值Tensor的字典
    all_labels : dict
        任务名称到真实值Tensor的字典
    """
    
    plt.rcParams.update({
    # 'font.size': 14,           # 全局字体大小
    'axes.titlesize': 14,      # 坐标轴标题字体大小
    'axes.labelsize': 14,      # 坐标轴标签字体大小
    # 'xtick.labelsize': 12,     # x轴刻度标签字体大小
    # 'ytick.labelsize': 12,     # y轴刻度标签字体大小
    'legend.fontsize': 10,     # 图例字体大小
    # 'figure.titlesize': 14,    # 图形标题字体大小
    })
    
    # 准备数据
    all_predictions = []
    all_ground_truth = []
    task_colors = []
    task_names = []
    
    # 为每个任务分配颜色
    cmap = plt.cm.get_cmap('tab20', len(all_preds))
    task_color_map = {}
    
    for idx, task in enumerate(all_preds.keys()):
        task_color_map[task] = cmap(idx)
    
    # 收集所有任务的数据
    for task in all_preds.keys():
        if task in all_labels:
            # 转换为numpy数组
            preds_np = all_preds[task].cpu().numpy() if torch.is_tensor(all_preds[task]) else all_preds[task]
            labels_np = all_labels[task].cpu().numpy() if torch.is_tensor(all_labels[task]) else all_labels[task]
            
            all_predictions.extend(preds_np.flatten())
            all_ground_truth.extend(labels_np.flatten())
            task_colors.extend([task_color_map[task]] * len(preds_np))
            task_names.extend([task] * len(preds_np))
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    task_colors = np.array(task_colors)
    
    # 创建图形
    plt.figure()
    
    # 为每个任务绘制散点
    unique_tasks = list(all_preds.keys())
    for task in unique_tasks:
        # 获取该任务的数据点索引
        task_indices = [i for i, t in enumerate(task_names) if t == task]
        if task_indices:
            plt.scatter(all_predictions[task_indices], 
                       all_ground_truth[task_indices], 
                       c=task_color_map[task], 
                       label=task, 
                       alpha=0.6, 
                       s=30,
                       edgecolors='w', 
                       linewidths=0.5)
    
    # 绘制y=x参考线
    min_val = min(all_predictions.min(), all_ground_truth.min())
    max_val = max(all_predictions.max(), all_ground_truth.max())
    margin = 0.1 * (max_val - min_val)
    
    plt.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'k--', alpha=0.7, label='y=x')
    
    # 计算每个任务的RMSE和R²，然后求平均
    task_rmse = []
    task_r2 = []
    
    for task in unique_tasks:
        if task in all_labels:
            # 转换为numpy数组
            preds_np = all_preds[task].cpu().numpy() if torch.is_tensor(all_preds[task]) else all_preds[task]
            labels_np = all_labels[task].cpu().numpy() if torch.is_tensor(all_labels[task]) else all_labels[task]
            
            # 计算RMSE
            mse = mean_squared_error(labels_np.flatten(), preds_np.flatten())
            rmse = np.sqrt(mse)
            task_rmse.append(rmse)
            
            # 计算R²
            r2 = r2_score(labels_np.flatten(), preds_np.flatten())
            task_r2.append(r2)
    
    # 计算平均指标
    avg_rmse = np.mean(task_rmse)
    avg_r2 = np.mean(task_r2)
    
    # 计算所有数据合并后的总体指标
    overall_rmse = np.sqrt(mean_squared_error(all_ground_truth, all_predictions))
    overall_r2 = r2_score(all_ground_truth, all_predictions)
    
    # 设置图形属性
    plt.xlabel('Predictions')
    plt.ylabel('Ground Truth')
    plt.title('Predictions vs Ground Truth')
    
    # 设置坐标轴范围
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    
    # 添加网格
    plt.grid()
    
    # 添加图例
    legend1 = plt.legend(loc='lower right')
    plt.gca().add_artist(legend1)
    
    # 添加指标文本
    metrics_text = f'Mean RMSE: {avg_rmse:.4f}\nMean R$^2$: {avg_r2:.4f}\nOverall RMSE: {overall_rmse:.4f}\nOverall R$^2$: {overall_r2:.4f}'
    plt.text(0.02, 0.98, metrics_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'),
             fontsize=10)
    
    # 添加任务数量信息
    # plt.text(0.02, 0.02, f'Number of tasks: {len(unique_tasks)}', 
    #          transform=plt.gca().transAxes,
    #          verticalalignment='bottom',
    #          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual_scatter.png', dpi=300)


def plot_predictions_vs_actual_dist(all_preds, all_labels):
    """
    绘制预测值vs真实值直方图
    
    Parameters:
    -----------
    all_preds : dict
        任务名称到预测值Tensor的字典
    all_labels : dict
        任务名称到真实值Tensor的字典
    """
    
    plt.rcParams.update({
    # 'font.size': 14,           # 全局字体大小
    'axes.titlesize': 14,      # 坐标轴标题字体大小
    'axes.labelsize': 14,      # 坐标轴标签字体大小
    # 'xtick.labelsize': 12,     # x轴刻度标签字体大小
    # 'ytick.labelsize': 12,     # y轴刻度标签字体大小
    'legend.fontsize': 12,     # 图例字体大小
    # 'figure.titlesize': 14,    # 图形标题字体大小
    })

    # 合并所有任务的数据
    all_pred_values = []
    all_label_values = []

    for task in all_preds.keys():
        # 将Tensor转换为numpy数组并展平
        pred_np = all_preds[task].numpy().flatten()
        label_np = all_labels[task].numpy().flatten()
        all_pred_values.extend(pred_np)
        all_label_values.extend(label_np)

    # 转换为numpy数组
    pred_array = np.array(all_pred_values)
    label_array = np.array(all_label_values)

    # 2. 计算合适的bins
    # 使用所有数据的全局最小值和最大值
    all_data = np.concatenate([pred_array, label_array])
    data_min, data_max = all_data.min(), all_data.max()

    # bin宽度
    n_bins = 50  # 默认值

    bins = np.linspace(data_min, data_max, n_bins)

    # 3. 计算直方图
    pred_hist, bin_edges = np.histogram(pred_array, bins=bins, density=False)
    label_hist, _ = np.histogram(label_array, bins=bins, density=False)

    # 4. 计算重叠面积
    # 计算相似性
    pred_hist_norm = pred_hist / np.sum(pred_hist)
    label_hist_norm = label_hist / np.sum(label_hist)
    similarity = np.sum(np.sqrt(pred_hist_norm * label_hist_norm))
    # 计算重叠面积
    min_heights = np.minimum(pred_hist, label_hist)
    overlap = np.sum(min_heights) / np.sum(pred_hist + label_hist - min_heights)

    # 5. 绘制直方图
    plt.figure()

    # 绘制预测值直方图
    plt.hist(pred_array, bins=bins, alpha=0.5, label='Predictions', 
            color='#1f77b4', linewidth=0.5)

    # 绘制真实值直方图
    plt.hist(label_array, bins=bins, alpha=0.5, label='Ground Truth', 
            color='#ff7f0e', linewidth=0.5)

    # 6. 添加重叠面积信息
    plt.text(0.02, 0.98, f'Overlap: {overlap:.4f}\nSimilarity: {similarity:.4f}', 
            transform=plt.gca().transAxes, 
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'),
            fontsize=12)

    # 7. 添加图例
    plt.legend(loc='lower right')

    # 8. 设置标签和标题
    plt.xlabel('Toxicity (-log(mol/kg))')
    plt.ylabel('Samples')
    plt.title('Predictions vs Ground Truth')

    # 9. 美化图表
    plt.grid()
    plt.tight_layout()

    plt.savefig('predictions_vs_actual_dist.png', dpi=300)


def plot_tsne_scatter(all_embds, all_labels):
    """
    为每个任务绘制t-SNE散点图
    
    参数:
    - all_embds: 包含每个任务嵌入向量的字典
    - all_labels: 包含每个任务标签的字典
    """

    plt.rcParams.update({
    # 'font.size': 14,           # 全局字体大小
    'axes.titlesize': 14,      # 坐标轴标题字体大小
    'axes.labelsize': 14,      # 坐标轴标签字体大小
    # 'xtick.labelsize': 12,     # x轴刻度标签字体大小
    # 'ytick.labelsize': 12,     # y轴刻度标签字体大小
    'legend.fontsize': 10,     # 图例字体大小
    'figure.titlesize': 14,    # 图形标题字体大小
    })

    all_embds['all'] = np.vstack(list(all_embds.values()))
    all_labels['all'] = np.vstack(list(all_labels.values()))

    for task_name in all_embds.keys():
        # 获取当前任务的嵌入和标签
        embeddings = all_embds[task_name]
        labels = all_labels[task_name]
        
        # 确保是numpy数组
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy().flatten()  # 展平为1D数组
        else:
            labels = labels.flatten()
        
        from sklearn.manifold import TSNE
        # 应用t-SNE降维
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            )
        
        # 降维到2D
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 创建图形
        plt.figure()
        
        # 绘制散点图，按标签值着色
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            # cmap='viridis',  # 可以使用其他颜色映射，如'plasma', 'coolwarm', 'Spectral'
            alpha=0.7,
            s=30,  # 点的大小
            # edgecolors='k',  # 点边缘颜色
            linewidths=0.5
        )
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, pad=0.03)
        cbar.set_label('Toxicity (-log(mol/kg))')
        
        # 添加标题和标签
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('{}'.format(task_name))
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图形
        plt.savefig('tsne_{}.png'.format(task_name), dpi=300)


def plot_uncertainty_kde(all_uncertainties):

    plt.rcParams.update({
    # 'font.size': 14,           # 全局字体大小
    'axes.titlesize': 14,      # 坐标轴标题字体大小
    'axes.labelsize': 14,      # 坐标轴标签字体大小
    # 'xtick.labelsize': 12,     # x轴刻度标签字体大小
    # 'ytick.labelsize': 12,     # y轴刻度标签字体大小
    'legend.fontsize': 10,     # 图例字体大小
    'figure.titlesize': 14,    # 图形标题字体大小
    })

    # 准备数据
    task_data = {}
    valid_tasks = []
    all_flattened_uncertainties = []

    for task_name, uncertainties in all_uncertainties.items():
        if isinstance(uncertainties, torch.Tensor):
            uncertainties_np = uncertainties.detach().cpu().numpy()
        else:
            uncertainties_np = np.array(uncertainties)

        task_data[task_name] = uncertainties_np.flatten()
        valid_tasks.append(task_name)
        all_flattened_uncertainties.append(uncertainties_np.flatten())

    if all_flattened_uncertainties:
        valid_tasks.append('all')
        concatenated_uncertainties = np.concatenate(all_flattened_uncertainties)
        task_data['all'] = concatenated_uncertainties

    n_tasks = len(valid_tasks)

    # 3. 创建图形和子图
    fig_width = 6.4
    fig_height = 2.4 * n_tasks

    fig, axes = plt.subplots(n_tasks, 1, figsize=(fig_width, fig_height), sharex=True)
    
    cmap = plt.colormaps['tab20']
    colors = cmap(np.linspace(0, 1, n_tasks))

    # 6. 为每个任务绘制子图
    for idx, (task_name, ax, color) in enumerate(zip(valid_tasks, axes, colors)):
        data = task_data[task_name]
        
        # 计算当前任务的统计信息
        task_min = data.min()
        task_max = data.max()
        task_mean = data.mean()
        task_std = data.std()
        
        # 计算KDE
        # 使用Silverman规则自动选择带宽
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method='silverman')
        
        # 设置x轴范围（当前任务的±10%范围）
        x_grid = np.linspace(-0.10, 0.20, 100)
        
        # 评估KDE
        y = kde(x_grid)
        
        # 绘制KDE曲线
        ax.plot(x_grid, y, color=color, linewidth=1, alpha=0.8)
        
        # 填充KDE曲线下方
        ax.fill_between(x_grid, y, alpha=0.3, color=color)
        
        # 添加统计信息文本
        stats_text = (f"$\mu$={task_mean:.3f} $\sigma$={task_std:.3f}")
            
        ax.text(0.05, 0.25, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', pad=0.3))
        
        # 设置子图标题和标签
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xticks([]) # 设置x轴刻度为空
        ax.set_yticks([]) # 设置y轴刻度为空

        # if idx == n_tasks-1:  # 只有最后的子图显示x轴标签
        #     ax.set_xlabel('Uncertainty', fontsize=9)
        ax.set_ylabel(task_name, rotation=0, fontsize=9)

        # 添加网格
        # ax.grid(True, alpha=0.3, linestyle='--')
        

    # 7. 设置整体图形标题
    # fig.suptitle('Uncertainty by Species')

    plt.tight_layout()
    plt.savefig('uncertainty.png', dpi=300)
