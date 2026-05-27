import numpy as np
import torch
import torch.nn as nn


class BCELoss():
    r"""The BCE loss founction.
    """
    def __init__(self):
        self.record = []
        self.bs = []
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        loss = torch.mean(loss)
        return loss

    def compute_sample_loss(self, pred, gt):
        r"""返回每个样本的损失，用于LearningLoss方法
        """
        loss = self.loss_fn(pred, gt)
        return loss
    
    def update_loss(self, pred, gt):
        r"""
        """
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss
    
    def update_mdn_loss(self, pred, gt):
        r"""
        """
        pi, mu, var = pred

        # 计算每个高斯分量的概率密度
        # 高斯概率密度: 1/√(2πσ²) * exp(-(y-μ)²/(2σ²))
        density_list = []
        for k in range(pi.size(1)):
            mu_k = mu[:, k, :]  # [batch_size, 1]
            var_k = var[:, k, :]  # [batch_size, 1]
            
            # 计算高斯概率密度 - 修正括号错误
            coefficient = 1.0 / torch.sqrt(2 * torch.tensor(np.pi) * var_k)  # 修正括号
            exponent = -((gt - mu_k) ** 2) / (2 * var_k)
            density_k = coefficient * torch.exp(exponent)  # [batch_size, 1]
            density_list.append(density_k.view(-1))  # [batch_size]
        
        # 堆叠所有分量的概率密度
        densities = torch.stack(density_list, dim=1)  # [batch_size, k]
        
        # 加权求和: sum_k π_k * N(y|μ_k, σ_k²)
        weighted_sum = (pi * densities).sum(dim=1)  # [batch_size]
        
        # 避免数值下溢，添加小的epsilon
        weighted_sum = torch.clamp(weighted_sum, min=1e-8)
        
        # 负对数: -log(weighted_sum)
        loss = -torch.log(weighted_sum).mean()  # [batch_size]

        self.record.append(loss.item())
        self.bs.append(mu.size()[0])
        return loss

    def average_loss(self):
        r"""
        """
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def reinit(self):
        r"""
        """
        self.record = []
        self.bs = []


class MSELoss():
    r"""The MSE loss founction.
    """
    def __init__(self):
        self.record = []
        self.bs = []
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        loss = torch.mean(loss)
        return loss
    
    def compute_sample_loss(self, pred, gt):
        r"""返回每个样本的损失，用于LearningLoss方法
        """
        loss = self.loss_fn(pred, gt)
        return loss
    
    def update_loss(self, pred, gt):
        r"""
        """
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss

    # def update_mdn_loss(self, pred, gt):
    #     r"""
    #     -log(pi*pred)
    #     """
    #     pi, mu, var = pred

    #     # 计算每个高斯分量的概率密度
    #     # 高斯概率密度: 1/√(2πσ²) * exp(-(y-μ)²/(2σ²))
    #     density_list = []
    #     for k in range(pi.size(1)):
    #         mu_k = mu[:, k, :]  # [batch_size, 1]
    #         var_k = var[:, k, :]  # [batch_size, 1]
            
    #         # 计算高斯概率密度 - 修正括号错误
    #         coefficient = 1.0 / torch.sqrt(2 * torch.tensor(np.pi) * var_k)  # 修正括号
    #         exponent = -((gt - mu_k) ** 2) / (2 * var_k)
    #         density_k = coefficient * torch.exp(exponent)  # [batch_size, 1]
    #         density_list.append(density_k.view(-1))  # [batch_size]
        
    #     # 堆叠所有分量的概率密度
    #     densities = torch.stack(density_list, dim=1)  # [batch_size, k]

    #     # 加权求和: sum_k π_k * N(y|μ_k, σ_k²)
    #     weighted_sum = (pi * densities).sum(dim=1)  # [batch_size]
        
    #     # 避免数值下溢，添加小的epsilon
    #     weighted_sum = torch.clamp(weighted_sum, min=1e-8)
        
    #     # 负对数: -log(weighted_sum)
    #     loss = -torch.log(weighted_sum).mean()  # [batch_size]

    #     self.record.append(loss.item())
    #     self.bs.append(mu.size()[0])
    #     return loss

    def update_mdn_loss(self, pred, gt):
        r"""
        pi * -log(pred)
        """
        pi, mu, var = pred

        # 计算每个高斯分量的负对数似然
        nll_list = []
        for k in range(pi.size(1)):
            mu_k = mu[:, k, :]  # [batch_size, 1]
            var_k = var[:, k, :]  # [batch_size, 1]
            
            # 计算高斯概率密度
            coefficient = 1.0 / torch.sqrt(2 * torch.tensor(np.pi) * var_k)
            exponent = -((gt - mu_k) ** 2) / (2 * var_k)
            density_k = coefficient * torch.exp(exponent)  # [batch_size, 1]
            
            # 避免数值下溢，添加小的epsilon
            density_k = torch.clamp(density_k, min=1e-8)

            # 计算负对数似然
            nll_k = -torch.log(density_k)  # [batch_size, 1]
            nll_list.append(nll_k.view(-1))  # [batch_size]

        # 堆叠所有分量的负对数似然
        nlls = torch.stack(nll_list, dim=1)  # [batch_size, k]

        # 加权求和: sum_k π_k * (-log(N(y|μ_k, σ_k²)))
        weighted_nll = (pi * nlls).sum(dim=1)  # [batch_size]
        
        # 取平均
        loss = weighted_nll.mean()

        self.record.append(loss.item())
        self.bs.append(mu.size()[0])
        return loss
    
    def average_loss(self):
        r"""
        """
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record*bs).sum()/bs.sum()
    
    def reinit(self):
        r"""
        """
        self.record = []
        self.bs = []