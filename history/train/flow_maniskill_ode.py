#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('./flow_grpo')
sys.path.append('./flow_grpo/diffusers_patch')
sys.path.append('./external/models')
sys.path.append('./external')

import os
# 设置wandb环境变量
os.environ["WANDB_ENTITY"] = "545242836-zju"
os.environ["WANDB_PROJECT"] = "Flow"
# 确保wandb在线模式
os.environ["WANDB_MODE"] = "online"

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from unet import ConditionalUnet1D
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from termcolor import colored
import pathlib
from skvideo.io import vwrite
import h5py
import json
import wandb  # 导入wandb

# 导入ManiSkill相关模块
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

# 导入Robot SDE Pipeline
from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import RobotActionPipeline, ActionDecoder, pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from external.models.maniskill_dataset import ManiSkillDataset


def extract_env_info_from_json(json_path):
    """从JSON文件中提取环境信息"""
    import json
    import os
    
    # 检查路径是否存在且是文件
    if os.path.exists(json_path) and os.path.isfile(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            env_info = data.get('env_info', {})
            env_id = env_info.get('env_id', None)
            env_kwargs = env_info.get('env_kwargs', {})
            
            return env_id, env_kwargs
        except Exception as e:
            print(f"读取JSON文件出错: {e}")
            return None, {}
    else:
        print(f"JSON文件不存在或不是文件: {json_path}")
        return None, {}


def compute_log_prob(action_unet, pipeline, sample, j, obs_cond, config=None):
    """
    计算机器人动作生成中SDE步骤的对数概率
    
    参数:
        action_unet: ConditionalUnet1D模型
        pipeline: RobotActionPipeline管线
        sample: 包含latents、timesteps、next_latents等的样本数据
        j: 当前时间步索引
        obs_cond: 观测状态条件 (batch_size, obs_dim)
        config: 配置参数（可选）
    
    返回:
        prev_sample: 上一步的样本
        log_prob: 对数概率
        prev_sample_mean: 上一步样本的均值
        std_dev_t: 标准差
    """
    
    # 获取当前时间步的数据
    current_latents = sample["latents"][:, j]  # (batch_size, action_horizon, action_dim)
    current_timestep = sample["timesteps"][:, j]  # (batch_size,)
    
    # 确保设备匹配
    device = current_latents.device
    current_timestep = current_timestep.to(device)
    
    # 使用ConditionalUnet1D进行噪声预测
    # 注意：这里不需要CFG（分类器自由引导），因为我们使用观测状态作为条件
    noise_pred = action_unet(
        sample=current_latents,
        timestep=current_timestep,
        global_cond=obs_cond
    )
    
    # # 如果配置中启用了CFG（可选功能，用于未来扩展）
    # if config is not None and hasattr(config, 'train') and hasattr(config.train, 'cfg') and config.train.cfg:
    #     # 对于机器人动作生成，我们可以使用零条件作为无条件预测
    #     batch_size = current_latents.shape[0]
    #     zero_cond = torch.zeros_like(obs_cond)
        
    #     # 预测无条件噪声
    #     noise_pred_uncond = action_unet(
    #         sample=current_latents,
    #         timestep=current_timestep,
    #         global_cond=zero_cond
    #     )
        
    #     # 应用分类器自由引导
    #     guidance_scale = getattr(config.sample, 'guidance_scale', 1.0)
    #     noise_pred = (
    #         noise_pred_uncond
    #         + guidance_scale * (noise_pred - noise_pred_uncond)
    #     )
    
    # 确保所有张量都在同一设备上
    next_latents = sample["next_latents"][:, j].to(device)
    
    # 计算SDE步骤的对数概率
    # 这是核心部分：使用预测的噪声和真实的next_latents计算对数概率
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,                    # 调度器
        noise_pred.float(),                   # 模型预测的噪声
        current_timestep,                     # 当前时间步
        current_latents.float(),              # 当前latents
        prev_sample=next_latents.float(),     # 真实的下一步latents
    )
    
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def load_maniskill_dataset(dataset_path, obs_key='obs', pred_horizon=16, batch_size=64, num_workers=4, val_ratio=0.1):
    """
    从ManiSkill数据集中加载数据
    
    参数:
        dataset_path: 数据集路径
        obs_key: 观测数据键，默认为'obs'
        pred_horizon: 预测序列长度
        batch_size: 批量大小
        num_workers: 数据加载器工作进程数
        val_ratio: 验证集比例
    
    返回:
        dataset: ManiSkillDataset对象
        dataloader: 数据加载器
        env_id: 环境ID
        env_kwargs: 环境参数
        obs_dim: 观测维度
        action_dim: 动作维度
    """
    print(f"加载数据集: {dataset_path}")
    
    # 获取环境ID和参数
    env_id = None
    env_kwargs = {}
    
    # 尝试从数据集的JSON文件获取环境信息
    if os.path.isdir(dataset_path):
        # 如果是目录，查找目录中是否有JSON文件
        json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
        if json_files:
            json_path = os.path.join(dataset_path, json_files[0])
            env_id, env_kwargs = extract_env_info_from_json(json_path)
            if env_id:
                print(f"从JSON文件加载环境信息: {env_id}")
    else:
        # 如果是文件，尝试查找同名的JSON文件
        json_path = dataset_path.replace('.h5', '.json')
        if os.path.exists(json_path):
            env_id, env_kwargs = extract_env_info_from_json(json_path)
            if env_id:
                print(f"从JSON文件加载环境信息: {env_id}")
    
    # 如果仍然没有环境ID，使用默认值
    if env_id is None:
        env_id = "PickCube-v1"
        # 为PickCube环境设置默认参数
        env_kwargs = {
            'sim_backend': 'physx_cpu',
            'control_mode': 'pd_ee_delta_pos',
            'obs_mode': 'state',
            'render_mode': 'rgb_array',
            'reward_mode': 'dense'
        }
        print(f"未指定环境ID，使用默认值: {env_id}")
        print(f"默认环境参数: {env_kwargs}")
    
    # 创建数据集
    dataset = ManiSkillDataset(
        dataset_dir=dataset_path,
        horizon=pred_horizon,
        pad_before=0,
        pad_after=0,
        val_ratio=val_ratio,
        obs_key=obs_key  # 使用指定的观测键
    )
    
    # 获取动作维度和观测维度
    action_dim = dataset.replay_buffer['action'][0].shape[-1]
    obs_dim = dataset.replay_buffer['obs'][0].shape[-1]
    
    print(f"数据集信息:")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  预测序列长度: {pred_horizon}")
    print(f"  验证集比例: {val_ratio}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"数据加载器创建完成:")
    print(f"  批量大小: {batch_size}")
    print(f"  工作进程数: {num_workers}")
    print(f"  批次数量: {len(dataloader)}")
    
    return dataset, dataloader, env_id, env_kwargs, obs_dim, action_dim



def train_model(
    dataset,
    dataloader,
    obs_dim,
    action_dim,
    action_horizon=16,
    learning_rate=1e-4,
    num_epochs=100,
    device="cuda",
    save_every=10,
    eval_every=5,
    checkpoint_dir="./checkpoints",
    log_interval=10,
    use_wandb=True  # 添加参数控制是否使用wandb
):
    """
    基于SDE的机器人动作生成模型训练
    
    参数:
        dataset: ManiSkillDataset数据集
        dataloader: 数据加载器
        obs_dim: 观测维度
        action_dim: 动作维度
        action_horizon: 动作序列长度
        learning_rate: 学习率
        num_epochs: 训练轮数
        device: 设备
        save_every: 保存模型间隔
        eval_every: 评估间隔
        checkpoint_dir: 检查点保存目录
        log_interval: 日志打印间隔
        use_wandb: 是否使用wandb记录训练过程
    
    返回:
        action_unet: 训练好的模型
        pipeline: 推理管线
        action_decoder: 训练好的动作解码器
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化wandb
    if use_wandb:
        try:
            print("正在初始化wandb...")
            wandb_login_success = wandb.login()
            if not wandb_login_success:
                print("警告: wandb登录失败，将禁用wandb记录")
                use_wandb = False
            else:
                wandb.init(
                    project="Flow",
                    entity="545242836-zju",
                    config={
                        "learning_rate": learning_rate,
                        "epochs": num_epochs,
                        "batch_size": dataloader.batch_size,
                        "action_dim": action_dim,
                        "obs_dim": obs_dim,
                        "action_horizon": action_horizon,
                        "architecture": "SDE-Flow-Matching",
                        "optimizer": "AdamW"
                    }
                )
                print("wandb初始化成功!")
        except Exception as e:
            print(f"wandb初始化失败: {e}")
            print("将禁用wandb记录")
            use_wandb = False
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. 初始化模型
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(device)
    
    # 2. 初始化调度器
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 3. 创建推理管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device
    )
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet  # 设置训练的模型
    
    # 创建action_decoder并设置为pipeline的属性
    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=256,
        num_layers=3
    ).to(device)
    pipeline.action_decoder = action_decoder
    
    # 使用wandb跟踪模型参数
    # 搞笑呢，训练速度直接降低8倍
    # if use_wandb:
    #     try:
    #         wandb.watch(action_unet, log="all", log_freq=log_interval)
    #         wandb.watch(action_decoder, log="all", log_freq=log_interval)
    #     except Exception as e:
    #         print(f"wandb.watch失败: {e}")
    
    # 打印action_decoder参数数量
    print(f"ActionDecoder参数数量: {sum(p.numel() for p in action_decoder.parameters())}")
    print(f"ActionDecoder可训练参数数量: {sum(p.numel() for p in action_decoder.parameters() if p.requires_grad)}")

    # 4. 初始化优化器,将 action_decoder 加入优化器参数
    optimizer = torch.optim.AdamW(
        list(action_unet.parameters()) + list(action_decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-6
    )
    
    # 5. 学习率调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_epochs * len(dataloader)
    )
    
    # 6. EMA模型（指数移动平均）
    ema_unet = EMAModel(
        action_unet.parameters(),
        decay=0.9999,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75
    )
    
    # 为action_decoder也创建EMA
    ema_decoder = EMAModel(
        action_decoder.parameters(),
        decay=0.9999,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75
    )
    
    print(f"模型初始化完成:")
    print(f"  UNet参数数量: {sum(p.numel() for p in action_unet.parameters())}")
    print(f"  UNet可训练参数数量: {sum(p.numel() for p in action_unet.parameters() if p.requires_grad)}")
    print(f"  总参数数量: {sum(p.numel() for p in action_unet.parameters()) + sum(p.numel() for p in action_decoder.parameters())}")
    print(f"  学习率: {learning_rate}")
    print(f"  总训练步数: {num_epochs * len(dataloader)}")
    
    # 7. 训练循环
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        action_unet.train()
        action_decoder.train()  # 确保action_decoder也处于训练模式
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取批次数据
            obs_seq = batch['obs'].to(device)      # (batch_size, sequence_length, obs_dim)
            action_seq = batch['action'].to(device) # (batch_size, sequence_length, action_dim)
            
            # 使用序列的第一个观测作为条件
            obs_cond = obs_seq[:, 0, :]  # (batch_size, obs_dim)
            target_actions = action_seq   # (batch_size, action_horizon, action_dim)
            
            batch_size = obs_cond.shape[0]
            
            # 1. 调用管线进行端到端训练
            # 创建自定义包装，确保梯度传递
            def generate_with_grad(pipeline, obs, **kwargs):
                # 修改pipeline中的pipeline_with_logprob函数装饰器
                original_torch_no_grad = torch.is_grad_enabled()
                torch.set_grad_enabled(True)
                
                # 调用管线函数（不使用@torch.no_grad()装饰器）
                result = pipeline_with_logprob(pipeline, obs=obs, **kwargs)
                
                # 恢复原始梯度设置
                torch.set_grad_enabled(original_torch_no_grad)
                return result
            
            # 使用自定义包装函数替代pipeline.generate_actions
            # 生成初始噪声并确保它需要梯度
            initial_noise = torch.randn(batch_size, action_horizon, action_dim, device=device, requires_grad=True)
            
            result = generate_with_grad(
                pipeline,
                obs=obs_cond,
                num_inference_steps=10,  # 减少步数加快训练
                output_type="tensor",
                return_dict=True,
                determistic=False,
                latents=initial_noise,  # 使用需要梯度的噪声
                action_decoder=action_decoder  # 传入同一个 action_decoder 实例
            )
                
            # 获取生成的动作
            pred_actions = result["actions"]
                
            # 2. 打印形状信息（首次迭代）
            if batch_idx == 0 and epoch == 0:
                print(f"目标动作形状: {target_actions.shape}")
                print(f"预测动作形状: {pred_actions.shape}")
                
            # 3. 计算损失（确保维度匹配）
            # 如果维度不匹配，可能需要调整
            if pred_actions.shape != target_actions.shape:
                if pred_actions.shape[1:] != target_actions.shape[1:]:
                    # 如果特征维度不匹配，可能需要修改pipeline
                    print(f"警告：特征维度不匹配 - 预测:{pred_actions.shape}, 目标:{target_actions.shape}")
                    # 尝试调整维度
                    if pred_actions.dim() == 2 and target_actions.dim() == 3:
                        # 如果预测是(batch_size, action_dim)而目标是(batch_size, horizon, action_dim)
                        # 取目标的第一个时间步
                        target_actions = target_actions[:, 0, :]
                    elif target_actions.dim() == 2 and pred_actions.dim() == 3:
                        # 如果目标是(batch_size, action_dim)而预测是(batch_size, horizon, action_dim)
                        # 取预测的第一个时间步
                        pred_actions = pred_actions[:, 0, :]
                
                # 4. 计算MSE损失
                loss = nn.functional.mse_loss(pred_actions, target_actions)
            else:
                # 如果维度匹配，直接计算MSE损失
                loss = nn.functional.mse_loss(pred_actions, target_actions)
            
            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(action_unet.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(action_decoder.parameters(), max_norm=1.0)  # 也对action_decoder梯度裁剪
            optimizer.step()
            lr_scheduler.step()
            
            # 6. 更新EMA
            ema_unet.step(action_unet.parameters())
            ema_decoder.step(action_decoder.parameters())  # 更新action_decoder的EMA
            
            # 7. 统计和记录
            epoch_loss += loss.item()
            global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                'Step': global_step
            })
            
            # 记录到wandb
            if use_wandb:
                try:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (batch_idx + 1) / len(dataloader),
                        "train/global_step": global_step
                    })
                except Exception as e:
                    print(f"wandb.log失败: {e}")
                    if batch_idx == 0:  # 只在第一个批次出错时打印一次
                        print("后续wandb日志记录将被禁用")
                        use_wandb = False
            
            # 定期日志记录
            if global_step % log_interval == 0:
                print(f"Step {global_step}: Loss={loss.item():.6f}, LR={lr_scheduler.get_last_lr()[0]:.2e}")
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        
        # 记录每个epoch的平均损失到wandb
        if use_wandb:
            try:
                wandb.log({
                    "train/epoch_avg_loss": avg_loss,
                    "train/epoch": epoch + 1
                })
            except Exception as e:
                if use_wandb:  # 只在第一次出错时打印
                    print(f"wandb.log失败: {e}")
                    print("后续wandb日志记录将被禁用")
                    use_wandb = False
        
        # 定期评估
        if (epoch + 1) % eval_every == 0:
            # 复制EMA权重到评估模型
            ema_unet.copy_to(action_unet.parameters())
            ema_decoder.copy_to(action_decoder.parameters())
            
            eval_loss = evaluate_model(action_unet, pipeline, dataloader, device, action_decoder=action_decoder, use_wandb=use_wandb)
            print(f"Epoch {epoch+1} 评估损失: {eval_loss:.6f}")
            
            # 记录评估损失到wandb
            if use_wandb:
                try:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/epoch": epoch + 1
                    })
                except Exception as e:
                    if use_wandb:  # 只在第一次出错时打印
                        print(f"wandb.log失败: {e}")
                        print("后续wandb日志记录将被禁用")
                        use_wandb = False
            
            # 保存最佳模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(
                    action_unet,
                    optimizer,
                    lr_scheduler,
                    ema_unet,
                    action_decoder,  # 保存action_decoder
                    epoch,
                    best_loss,
                    os.path.join(checkpoint_dir, "best_model.pth")
                )
                print(f"保存最佳模型，损失: {best_loss:.6f}")
                
                # 记录最佳模型信息到wandb
                if use_wandb:
                    try:
                        wandb.run.summary["best_loss"] = best_loss
                        wandb.run.summary["best_epoch"] = epoch + 1
                        
                        # 将最佳模型保存到wandb
                        wandb.save(os.path.join(checkpoint_dir, "best_model.pth"))
                    except Exception as e:
                        if use_wandb:  # 只在第一次出错时打印
                            print(f"wandb保存模型失败: {e}")
                            print("后续wandb日志记录将被禁用")
                            use_wandb = False
        
        # 定期保存检查点
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(
                action_unet,
                optimizer,
                lr_scheduler,
                ema_unet,
                action_decoder,  # 保存action_decoder
                epoch,
                avg_loss,
                checkpoint_path
            )
            print(f"保存检查点: epoch_{epoch+1}")
            
            # 将检查点保存到wandb
            if use_wandb:
                try:
                    wandb.save(checkpoint_path)
                except Exception as e:
                    if use_wandb:  # 只在第一次出错时打印
                        print(f"wandb保存检查点失败: {e}")
                        print("后续wandb日志记录将被禁用")
                        use_wandb = False
    
    print("训练完成！")
    
    # 关闭wandb
    if use_wandb and wandb.run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"wandb.finish()失败: {e}")
    
    return action_unet, pipeline, action_decoder


def evaluate_model(action_unet, pipeline, dataloader, device, num_samples=100, action_decoder=None, use_wandb=False):
    """
    评估模型性能
    """
    action_unet.eval()
    if action_decoder is not None:
        action_decoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * batch['obs'].shape[0] >= num_samples:
                break
                
            # 处理数据格式
            obs_seq = batch['obs'].to(device)
            action_seq = batch['action'].to(device)
            
            obs_cond = obs_seq[:, 0, :]  # (batch_size, obs_dim)
            true_actions = action_seq    # (batch_size, action_horizon, action_dim)
            
            batch_size = obs_cond.shape[0]
            
            # 使用管线生成动作
            from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import pipeline_with_logprob
            
            # 调用管线函数
            result = pipeline_with_logprob(
                pipeline,
                obs=obs_cond,
                num_inference_steps=10,  # 使用与训练相同的步数
                output_type="tensor",
                return_dict=True,
                determistic=False,  # 评估时使用确定性生成
                latents=torch.randn(batch_size, true_actions.shape[1], true_actions.shape[2]).to(device),
                action_decoder=action_decoder  # 传入相同的action_decoder
            )
            
            predicted_actions = result["actions"]
            
            # 确保维度匹配
            if predicted_actions.shape != true_actions.shape:
                if predicted_actions.dim() == 2 and true_actions.dim() == 3:
                    true_actions = true_actions[:, 0, :]
                elif true_actions.dim() == 2 and predicted_actions.dim() == 3:
                    predicted_actions = predicted_actions[:, 0, :]
            
            # 计算MSE损失
            loss = nn.functional.mse_loss(predicted_actions, true_actions)
            total_loss += loss.item()
            num_batches += 1
            
            # 打印第一个批次的结果
            if batch_idx == 0:
                print(f"评估 - 预测动作形状: {predicted_actions.shape}, 真实动作形状: {true_actions.shape}")
    
    action_unet.train()
    if action_decoder is not None:
        action_decoder.train()
    
    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, lr_scheduler, ema_model, action_decoder, epoch, loss, path):
    """
    保存检查点
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'action_decoder_state_dict': action_decoder.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, lr_scheduler, ema_model, action_decoder, path):
    """
    加载检查点
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    
    # 如果检查点包含action_decoder状态，则加载
    if 'action_decoder_state_dict' in checkpoint:
        action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

def test_trained_model(action_unet, pipeline, test_obs, env_id, env_kwargs):
    """
    测试训练好的模型
    """
    print(f"测试训练好的模型...")
        
    batch_size = test_obs.shape[0]
    action_horizon = pipeline.action_horizon
    action_dim = pipeline.action_dim
    
    # 调用管线函数
    result = pipeline_with_logprob(
        pipeline,
        obs=test_obs,
        num_inference_steps=20,  # 测试时使用更多步骤提高质量
        output_type="clipped",
        return_dict=True,
        determistic=True,
        latents=torch.randn(batch_size, action_horizon, action_dim).to(test_obs.device)
    )
    
    generated_actions = result["actions"]
    
    print(f"生成动作序列形状: {generated_actions.shape}")
    print(f"动作范围: [{generated_actions.min():.3f}, {generated_actions.max():.3f}]")
    
    # 可视化动作序列
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    for i in range(min(4, generated_actions.shape[0])):  # 最多显示4个样本
        plt.subplot(2, 2, i+1)
        actions_cpu = generated_actions[i].cpu().numpy()
        plt.plot(actions_cpu)
        plt.title(f"Sample {i+1} Action Sequence")
        plt.xlabel("Time Step")
        plt.ylabel("Action Value")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("generated_actions.png")
    plt.show()
    
    return generated_actions


def test_model_in_env(action_unet, checkpoint_path, obs_dim, action_dim, action_horizon, env_id, env_kwargs=None, device="cuda", num_inference_steps=20, render=True, deterministic=True):
    """
    在ManiSkill环境中测试训练好的SDE流匹配模型
    
    参数:
        action_unet: 训练好的UNet模型
        checkpoint_path: 检查点路径
        obs_dim: 观测维度
        action_dim: 动作维度
        action_horizon: 动作序列长度
        env_id: ManiSkill环境ID
        env_kwargs: 环境参数
        device: 计算设备
        num_inference_steps: 推理步数
        render: 是否渲染环境
        deterministic: 是否使用确定性生成
    """
    import gymnasium as gym
    import mani_skill.envs
    import collections
    import numpy as np
    from mani_skill.utils.wrappers.record import RecordEpisode
    from flow_grpo.diffusers_patch.robot_sde_pipeline_with_logprob_unet import RobotActionPipeline, ActionDecoder
    
    print(f"\n{'='*60}")
    print(f"在ManiSkill环境中测试模型")
    print(f"{'='*60}")
    
    # 创建action_decoder
    action_decoder = ActionDecoder(
        action_dim=action_dim,
        action_horizon=action_horizon,
        hidden_dim=256,
        num_layers=3
    ).to(device)
    
    # 加载模型权重
    if checkpoint_path:
        print(f"加载检查点: {checkpoint_path}")
        # 首先将模型加载到CPU上，然后再转移到目标设备
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        action_unet.load_state_dict(checkpoint['model_state_dict'])
        
        # 如果检查点包含action_decoder状态，则加载
        if 'action_decoder_state_dict' in checkpoint:
            print("从检查点加载action_decoder权重")
            action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
        else:
            print("警告：检查点不包含action_decoder权重，使用随机初始化的action_decoder")
    
    # 确保模型在正确的设备上
    print(f"将模型移动到设备: {device}")
    action_unet = action_unet.to(device)
    action_decoder = action_decoder.to(device)
    action_unet.eval()
    action_decoder.eval()

    # 防止梯度计算
    for param in action_unet.parameters():
        param.requires_grad = False
    for param in action_decoder.parameters():
        param.requires_grad = False
    
    # 创建管线
    pipeline = RobotActionPipeline(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        device=device
    )
    
    # 设置调度器
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.setup_scheduler(scheduler)
    pipeline._action_unet = action_unet
    pipeline.action_decoder = action_decoder  # 设置action_decoder
    
    # 设置默认环境参数
    if env_kwargs is None:
        env_kwargs = {
            'sim_backend': 'physx_cpu',
            'control_mode': 'pd_ee_delta_pos',
            'obs_mode': 'state',
            'render_mode': 'rgb_array',
            'reward_mode': 'dense'
        }
    
    # 创建ManiSkill环境
    print(f"初始化环境：{env_id}")
    env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
    
    # 创建视频保存目录
    video_dir = "./videos"
    os.makedirs(video_dir, exist_ok=True)
    
    # 测试参数
    max_steps = 500
    test_start_seed = 10000
    n_test = 3
    
    # 设置录制参数
    if render:
        record_kwargs = {
            'save_trajectory': False,
            'save_video': True,
            'video_fps': 30
        }
        env = RecordEpisode(env, video_dir, **record_kwargs)
    
    # 运行多次测试
    rewards_all = []
    for test_idx in range(n_test):
        seed = test_start_seed + test_idx
        obs, info = env.reset(seed=seed)
        
        # 初始化观测
        # 检查obs是否已经是Tensor类型
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(device, dtype=torch.float32)
            # 确保批次维度存在
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        else:
            # 如果是numpy数组或其他类型，转换为Tensor
            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        
        # 记录奖励和步数
        rewards = []
        done = False
        step_idx = 0
        
        # 运行测试
        with tqdm(total=max_steps, desc=f"测试 {env_id} (seed={seed})") as pbar:
            while not done and step_idx < max_steps:
                # 使用模型预测动作序列
                with torch.no_grad():
                    # 调用管线生成动作，确保使用相同的action_decoder
                    result = pipeline_with_logprob(
                        pipeline,
                        obs=obs_tensor,
                        num_inference_steps=num_inference_steps,
                        output_type="clipped",  # 输出范围限制在[-1, 1]
                        return_dict=True,
                        determistic=deterministic,
                        action_decoder=action_decoder  # 使用加载的action_decoder
                    )
                    
                    # 获取生成的动作序列
                    actions = result["actions"][0].cpu().numpy()  # 取第一个样本
                    
                    # 只执行action_horizon中的前几个动作
                    execute_horizon = min(4, action_horizon)  # 可以调整执行的动作数量
                    
                    # 执行生成的动作
                    for j in range(execute_horizon):
                        # 执行环境步骤
                        action = actions[j].copy()  # 确保获取一个副本，避免修改原始数组
                        
                        # 确保动作类型正确
                        if not isinstance(action, np.ndarray):
                            action = np.array(action, dtype=np.float32)
                            
                        # 执行一步
                        try:
                            obs, reward, terminated, truncated, info = env.step(action)
                        except Exception as e:
                            print(f"执行动作时出错: {e}")
                            print(f"动作类型: {type(action)}, 形状: {action.shape}, 值: {action}")
                            raise
                        
                        # 更新观测
                        # 检查obs是否已经是Tensor类型
                        if isinstance(obs, torch.Tensor):
                            obs_tensor = obs.to(device, dtype=torch.float32)
                            # 确保批次维度存在
                            if obs_tensor.dim() == 1:
                                obs_tensor = obs_tensor.unsqueeze(0)
                        else:
                            # 如果是numpy数组或其他类型，转换为Tensor
                            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                        
                        # 记录奖励
                        rewards.append(reward)
                        
                        # 更新进度条
                        step_idx += 1
                        pbar.update(1)
                        
                        # 确保reward是Python标量值
                        reward_value = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
                        total_reward = sum(r.item() if isinstance(r, torch.Tensor) else float(r) for r in rewards)
                        
                        pbar.set_postfix(reward=f"{reward_value:.3f}", total=f"{total_reward:.3f}")
                        
                        # 检查是否结束
                        if terminated or step_idx >= max_steps:
                            done = True
                            break
        
        # 计算总奖励和平均奖励（确保转换为Python标量）
        total_reward = sum(r.item() if isinstance(r, torch.Tensor) else float(r) for r in rewards)
        avg_reward = total_reward / len(rewards) if rewards else 0.0
        
        # 打印测试结果
        print(f"\n测试 {test_idx+1}/{n_test} 完成")
        print(f"总步数: {step_idx}")
        print(f"累计奖励: {total_reward:.3f}")
        print(f"平均奖励: {avg_reward:.3f}")
        
        # 将rewards转换为普通Python列表保存（避免后续处理中的Tensor问题）
        python_rewards = [r.item() if isinstance(r, torch.Tensor) else float(r) for r in rewards]
        rewards_all.append(python_rewards)
        
        # 保存视频
        if render:
            env.flush_video(ignore_empty_transition=False)
    
    # 关闭环境
    env.close()
    
    # 计算平均累计奖励（已确保是Python标量）
    avg_total_reward = sum([sum(r) for r in rewards_all]) / len(rewards_all)
    print(f"\n{n_test}次测试的平均累计奖励: {avg_total_reward:.3f}")
    
    return rewards_all


def main():
    """
    主函数
    """
    print("=" * 60)
    print("ManiSkill 真正的 SDE 模型训练")
    print("=" * 60)
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='ManiSkill SDE流匹配训练与评估')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载特定检查点')
    parser.add_argument('--env_id', type=str, default=None, help='环境ID，如果不指定则从数据集JSON中获取')
    parser.add_argument('--dataset', type=str, default=None, help='数据集路径')
    parser.add_argument('--gpu', type=int, default=1, help='使用的GPU ID')
    parser.add_argument('--steps', type=int, default=20, help='SDE推理步数')
    parser.add_argument('--deterministic', action='store_true', help='使用确定性生成')
    parser.add_argument('--use_wandb', action='store_true', help='启用wandb记录训练过程')
    args = parser.parse_args()
    
    # 配置参数
    config = {
        # 使用实际的ManiSkill数据集路径
        'dataset_path': args.dataset or '/home/lsx/rzx/flow/episodes',  # 实际ManiSkill数据路径
        'obs_key': 'obs',
        'pred_horizon': 16,
        'batch_size': 32,  # 减小batch size以适应内存
        'num_workers': 4,
        'val_ratio': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 100,  # 增加epochs进行充分训练
        'device': f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu',
        'save_every': 10,
        'eval_every': 5,
        'checkpoint_dir': './checkpoints/maniskill_pickcube',  # 使用特定任务名称
        'log_interval': 10,
        'only_show_model': False,  # 是否只显示模型结构而不训练
    }
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        # 设置当前CUDA设备
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用设备: {device}")
        print(f"当前使用GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print(f"使用设备: CPU")
    
    # 更新配置中的设备
    config['device'] = str(device)
    
    # 检查并创建必要的目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 检查数据集路径或提供备用路径
    possible_paths = [
        config['dataset_path'],
        '/home/lsx/rzx/flow/episodes',  # 确保实际路径在列表中
        './episodes',
        './data/episodes',
        os.path.expanduser('~/datasets/maniskill'),
        # 添加更多可能的路径
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None and args.train:
        print("❌ 未找到数据集路径，请检查以下路径:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\n请按照以下步骤操作:")
        print("1. 确保ManiSkill数据集存在于 /home/lsx/rzx/flow/episodes")
        print("2. 或者修改main()函数中的dataset_path配置")
        print("3. 或者将数据集放在上述路径之一")
        
        # 创建示例数据集用于测试
        print("\n正在创建示例数据集用于测试...")
        dataset_path = create_dummy_dataset(config['pred_horizon'])
        print(f"✓ 示例数据集创建完成: {dataset_path}")
        config['dataset_path'] = dataset_path
    elif dataset_path is not None:
        print(f"✓ 找到数据集路径: {dataset_path}")
        config['dataset_path'] = dataset_path
    
    # 环境ID
    env_id = args.env_id
    env_kwargs = None
    
    # 加载数据集和环境信息
    if args.train or (dataset_path is not None and not args.test):
        try:
            # 加载数据集
            print("加载数据集...")
            dataset, dataloader, env_id_from_data, env_kwargs, obs_dim, action_dim = load_maniskill_dataset(
                dataset_path=config['dataset_path'],
                obs_key=config['obs_key'],
                pred_horizon=config['pred_horizon'],
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                val_ratio=config['val_ratio']
            )
            
            print(f"✓ 数据集加载成功")
            print(f"  环境ID: {env_id_from_data}")
            print(f"  观测维度: {obs_dim}")
            print(f"  动作维度: {action_dim}")
            print(f"  动作序列长度: {config['pred_horizon']}")
            
            if not env_id:
                env_id = env_id_from_data
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            if args.train:
                return
    
    # 如果仍然没有环境ID，使用默认值
    if not env_id:
        env_id = "PickCube-v1"
        # 为PickCube环境设置默认参数
        env_kwargs = {
            'sim_backend': 'physx_cpu',
            'control_mode': 'pd_ee_delta_pos',
            'obs_mode': 'state',
            'render_mode': 'rgb_array',
            'reward_mode': 'dense'
        }
        print(f"未指定环境ID，使用默认值: {env_id}")
        print(f"默认环境参数: {env_kwargs}")
        obs_dim = 42  # 默认观测维度
        action_dim = 4  # 默认动作维度
    
    # 创建模型
    action_unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(device)
    
    # 显示模型结构
    if config['only_show_model']:
        show_model(action_unet)
        print("模型结构显示完成，程序退出。")
        return
    
    # 训练模型
    if args.train:
        try:
            print("\n开始训练...")
            action_unet, pipeline, action_decoder = train_model(
                dataset=dataset,
                dataloader=dataloader,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_horizon=config['pred_horizon'],
                learning_rate=config['learning_rate'],
                num_epochs=config['num_epochs'],
                device=config['device'],
                save_every=config['save_every'],
                eval_every=config['eval_every'],
                checkpoint_dir=config['checkpoint_dir'],
                log_interval=config['log_interval'],
                use_wandb=args.use_wandb  # 传入wandb参数
            )
            
            print("✓ 训练完成")
            print(f"  UNet参数数量: {sum(p.numel() for p in action_unet.parameters())}")
            print(f"  ActionDecoder参数数量: {sum(p.numel() for p in action_decoder.parameters())}")
        
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保在发生错误时也关闭wandb
            if args.use_wandb and wandb.run is not None:
                try:
                    wandb.finish()
                except:
                    pass
            
            return
    
    # 测试模型
    if args.test:
        try:
            print("\n开始在ManiSkill环境中测试模型...")
            
            # 确定检查点路径
            checkpoint_path = args.checkpoint
            if not checkpoint_path:
                # 查找最新的检查点
                checkpoints = sorted([f for f in os.listdir(config['checkpoint_dir']) if f.endswith('.pth')])
                if checkpoints:
                    checkpoint_path = os.path.join(config['checkpoint_dir'], checkpoints[-1])
                    print(f"使用最新检查点: {checkpoint_path}")
                else:
                    print("❌ 未找到检查点文件！")
                    return
            
            # 在环境中测试模型
            rewards = test_model_in_env(
                action_unet=action_unet,
                checkpoint_path=checkpoint_path,
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_horizon=config['pred_horizon'],
                env_id=env_id,
                env_kwargs=env_kwargs,
                device=device,
                num_inference_steps=args.steps,
                render=True,
                deterministic=args.deterministic
            )
            
            print("✓ 测试完成")
            
        except Exception as e:
            print(f"❌ 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return
    
    print("\n" + "=" * 60)
    print("🎉 程序完成！")
    if args.train:
        print(f"模型检查点保存在: {config['checkpoint_dir']}")
    if args.test:
        print(f"测试视频保存在: ./videos")
    print("=" * 60)
    
    # 确保程序结束时关闭wandb
    if args.use_wandb and wandb.run is not None:
        try:
            wandb.finish()
        except:
            pass

def test_compute_log_prob():
    """
    测试compute_log_prob函数
    """
    print("=" * 60)
    print("测试 compute_log_prob 函数")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置测试参数
    batch_size = 2
    obs_dim = 42
    action_dim = 8
    action_horizon = 16
    num_steps = 10
    
    print(f"测试参数:")
    print(f"  设备: {device}")
    print(f"  批量大小: {batch_size}")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  动作序列长度: {action_horizon}")
    print(f"  去噪步数: {num_steps}")
    
    try:
        # 创建模型和管线
        pipeline = RobotActionPipeline(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            device=device
        )
        
        # 设置调度器
        scheduler = FlowMatchEulerDiscreteScheduler()
        pipeline.setup_scheduler(scheduler)
        
        # 确保调度器的时间步在正确的设备上
        if hasattr(scheduler, 'timesteps') and scheduler.timesteps is not None:
            scheduler.timesteps = scheduler.timesteps.to(device)
        if hasattr(scheduler, 'sigmas') and scheduler.sigmas is not None:
            scheduler.sigmas = scheduler.sigmas.to(device)
        
        # 初始化UNet
        action_unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim
        ).to(device)
        
        # 创建测试数据
        obs_cond = torch.randn(batch_size, obs_dim).to(device)
        
        # 创建样本数据（模拟从采样过程中获得的数据）
        latents = torch.randn(batch_size, num_steps, action_horizon, action_dim).to(device)
        next_latents = torch.randn(batch_size, num_steps, action_horizon, action_dim).to(device)
        timesteps = torch.randint(0, 1000, (batch_size, num_steps)).to(device)
        
        sample = {
            "latents": latents,
            "next_latents": next_latents,
            "timesteps": timesteps
        }
        
        print(f"\n创建测试数据:")
        print(f"  观测条件形状: {obs_cond.shape}")
        print(f"  latents形状: {sample['latents'].shape}")
        print(f"  next_latents形状: {sample['next_latents'].shape}")
        print(f"  timesteps形状: {sample['timesteps'].shape}")
        
        # 测试不同时间步的对数概率计算
        print(f"\n测试compute_log_prob函数...")
        log_probs = []
        
        for j in range(min(3, num_steps)):  # 只测试前3个时间步
            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                action_unet=action_unet,
                pipeline=pipeline,
                sample=sample,
                j=j,
                obs_cond=obs_cond,
                config=None
            )
            
            log_probs.append(log_prob)
            
            print(f"  时间步 {j}:")
            print(f"    prev_sample形状: {prev_sample.shape}")
            print(f"    log_prob形状: {log_prob.shape}")
            print(f"    log_prob范围: [{log_prob.min():.3f}, {log_prob.max():.3f}]")
            print(f"    log_prob均值: {log_prob.mean():.3f}")
            print(f"    prev_sample_mean形状: {prev_sample_mean.shape}")
            print(f"    std_dev_t形状: {std_dev_t.shape}")
        
        # 验证对数概率的合理性
        all_log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_tested_steps)
        
        print(f"\n对数概率统计:")
        print(f"  所有log_prob形状: {all_log_probs.shape}")
        print(f"  所有log_prob均值: {all_log_probs.mean():.3f}")
        print(f"  所有log_prob标准差: {all_log_probs.std():.3f}")
        print(f"  是否包含NaN: {torch.isnan(all_log_probs).any()}")
        print(f"  是否包含Inf: {torch.isinf(all_log_probs).any()}")
        
        # 测试是否可微分（用于梯度计算）
        print(f"\n测试梯度计算...")
        loss = all_log_probs.mean()
        loss.backward()
        
        has_gradients = any(p.grad is not None for p in action_unet.parameters() if p.requires_grad)
        print(f"  模型参数是否有梯度: {has_gradients}")
        
        if has_gradients:
            grad_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in action_unet.parameters() if p.grad is not None))
            print(f"  梯度范数: {grad_norm:.6f}")
        
        print(f"\n✓ compute_log_prob 函数测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    main()