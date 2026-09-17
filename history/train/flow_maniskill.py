#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
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
from termcolor import colored
import pathlib
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
from external.models.maniskill_dataset import ManiSkillDataset
import gymnasium as gym
import h5py
import wandb  # 导入wandb

# 导入ManiSkill相关模块
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode


def train_model(noise_pred_net, optimizer, lr_scheduler, dataloader, num_epochs, obs_horizon, device, FM, ema, checkpoint_dir, use_wandb=False):
    """
    训练Flow Matching模型
    
    参数:
        noise_pred_net: 模型网络
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        dataloader: 数据加载器
        num_epochs: 训练轮数
        obs_horizon: 观测序列长度
        device: 训练设备
        FM: Flow Matching对象
        ema: 指数移动平均模型
        checkpoint_dir: 检查点保存目录
        use_wandb: 是否使用wandb记录训练过程
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录训练损失
    loss_history = []
    global_step = 0
    
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            x_img = data['obs'][:, :obs_horizon].to(device)
            x_traj = data['action'].to(device)

            x_traj = x_traj.float()
            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

            # 编码观测特征
            obs_cond = x_img.flatten(start_dim=1)
            vt = noise_pred_net(xt, timestep, global_cond=obs_cond)

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # 更新EMA模型权重
            ema.step(noise_pred_net.parameters())
            
            # 更新全局步数
            global_step += 1
            
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

        # 计算平均损失
        avg_loss_train = total_loss_train / len(dataloader)
        loss_history.append(avg_loss_train.detach().cpu().numpy())
        
        print(colored(f"epoch: {epoch:>04d},  loss_train: {avg_loss_train:.8f}", 'yellow'))
        
        # 保存检查点
        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            ema.copy_to(noise_pred_net.parameters())
            checkpoint_path = os.path.join(checkpoint_dir, f'flow_ema_{epoch:05d}.pth')
            torch.save({
                'noise_pred_net': noise_pred_net.state_dict(),
                'epoch': epoch,
                'loss': avg_loss_train.item(),
            }, checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")
            
            # 将检查点保存到wandb
            if use_wandb:
                try:
                    wandb.save(checkpoint_path)
                except Exception as e:
                    if use_wandb:  # 只在第一次出错时打印
                        print(f"wandb保存检查点失败: {e}")
                        print("后续wandb日志记录将被禁用")
                        use_wandb = False
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))
    plt.close()
    
    # 关闭wandb
    if use_wandb and wandb.run is not None:
        try:
            wandb.finish()
        except Exception as e:
            print(f"wandb.finish()失败: {e}")
    
    return loss_history


def test_model(noise_pred_net, checkpoint_path, obs_horizon, action_horizon, device, pred_horizon, action_dim, env_id, env_kwargs=None, render=True):
    """
    测试训练好的模型
    
    参数:
        noise_pred_net: 模型网络
        checkpoint_path: 检查点路径
        obs_horizon: 观测序列长度
        action_horizon: 动作序列长度
        device: 设备
        pred_horizon: 预测序列长度
        action_dim: 动作维度
        env_id: ManiSkill环境ID
        env_kwargs: 环境参数
        render: 是否渲染环境
    """
    # 加载模型权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
    noise_pred_net.eval()
    
    # 设置默认环境参数
    if env_kwargs is None:
        env_kwargs = {}
    
    # 创建ManiSkill环境
    print(f"初始化环境：{env_id}")
    env = gym.make(env_id,reconfiguration_freq=1, **env_kwargs)
    
    # 创建视频保存目录
    os.makedirs("videos", exist_ok=True)
    
    # 测试参数
    max_steps = 500
    test_start_seed = 10000
    n_test = 3
    
    # 设置录制参数
    record_kwargs = {}
    if render:
        record_kwargs = {
            'save_trajectory': False,
            'save_video': True,
            'video_fps': 30
        }
        video_dir = "videos"
        os.makedirs(video_dir, exist_ok=True)
        env = RecordEpisode(env, video_dir, **record_kwargs)
    
    # 运行多次测试
    rewards_all = []
    for test_idx in range(n_test):
        seed = test_start_seed + test_idx
        obs, info = env.reset(seed=seed)
        
        # 初始化观测队列
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        # 记录奖励和步数
        rewards = []
        done = False
        step_idx = 0
        
        # 运行测试
        with tqdm(total=max_steps, desc=f"Eval {env_id} (seed={seed})") as pbar:
            while not done:
                # 准备观测数据
                x_img = np.stack([x for x in obs_deque])
                x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                
                # 使用模型预测动作
                with torch.no_grad():
                    # 获取观测特征
                    obs_cond = x_img.flatten(start_dim=1)
                    print(obs_cond.shape)

                    # 使用Flow Matching生成轨迹
                    time_horizon = 16
                    for i in range(time_horizon):
                        noise = torch.rand(1, pred_horizon, action_dim).to(device)
                        x0 = noise.expand(x_img.shape[0], -1, -1)
                        timestep = torch.tensor([i / time_horizon]).to(device)

                        if i == 0:
                            vt = noise_pred_net(x0, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / time_horizon + x0)
                        else:
                            vt = noise_pred_net(traj, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / time_horizon + traj)
                    
                    # 获取预测的动作序列
                    action_pred = traj.detach().to('cpu').numpy()[0]
                    
                    # 只使用action_horizon数量的动作
                    start = obs_horizon - 1
                    end = start + action_horizon
                    actions = action_pred[start:end, :]
                    
                    # 执行动作序列
                    for j in range(len(actions)):
                        # 执行环境步骤
                        obs, reward, terminated, truncated, info = env.step(actions[j])
                        # done = terminated or truncated 当抓到物体时，truncated 就会一直返回 True
                        done = terminated

                        # 更新观测队列和奖励
                        obs_deque.append(obs)
                        rewards.append(reward)
                        
                        # 更新进度条
                        step_idx += 1
                        pbar.update(1)
                        pbar.set_postfix(reward=reward)
                        
                        # 检查是否结束
                        if step_idx >= max_steps or done:
                            done = True
                            break
        
        # 打印测试结果
        print(f"测试 {test_idx+1}/{n_test} 完成")
        print(f"总步数: {step_idx}")
        print(f"累计奖励: {sum(rewards)}")
        print(f"平均奖励: {sum(rewards)/len(rewards) if rewards else 0}")
        rewards_all.append(rewards)
        
        # 保存视频
        if render:
            env.flush_video(ignore_empty_transition=False)
    
    # 关闭环境
    env.close()
    
    return rewards_all


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


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='ManiSkill Flow Matching训练与评估')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载特定检查点')
    parser.add_argument('--dataset', type=str, required=True, help='数据集路径')
    parser.add_argument('--env_id', type=str, default=None, help='环境ID，如果不指定则从数据集JSON中获取')
    parser.add_argument('--task_name', type=str, default=None, help='任务名称，用于保存模型和结果')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    parser.add_argument('--obs_key', type=str, default='obs', help='观测数据键，默认为obs')
    parser.add_argument('--use_wandb', action='store_true', help='启用wandb记录训练过程')
    args = parser.parse_args()
    
    # 初始化wandb
    use_wandb = args.use_wandb
    if use_wandb:
        try:
            print("正在初始化wandb...")
            wandb_login_success = wandb.login()
            if not wandb_login_success:
                print("警告: wandb登录失败，将禁用wandb记录")
                use_wandb = False
            else:
                print("wandb初始化成功!")
        except Exception as e:
            print(f"wandb初始化失败: {e}")
            print("将禁用wandb记录")
            use_wandb = False
    
    if args.train:
    # 设置设备
        torch.cuda.set_device(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        torch.cuda.set_device(1)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前使用GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
    
    # 参数设置
    dataset_path = args.dataset  # 数据集路径
    
    # 获取环境ID和参数
    env_id = args.env_id
    env_kwargs = {}
    
    # 如果未指定环境ID，尝试从数据集的JSON文件获取
    if env_id is None:
        # 检查是否是目录路径
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
    
    # 设置任务名称
    task_name = args.task_name or env_id
    
    # 模型参数
    obs_horizon = 1  # 观测序列长度
    pred_horizon = 16  # 预测序列长度
    action_horizon = 8  # 执行的动作序列长度
    
    # 训练参数
    batch_size = 64
    num_epochs = 500
    num_workers = 4
    
    # 创建数据集
    print(f"加载数据集: {dataset_path}")
    dataset = ManiSkillDataset(
        dataset_dir=dataset_path,
        horizon=pred_horizon,
        pad_before=0,
        pad_after=0,
        val_ratio=0.1,
        obs_key=args.obs_key  # 使用命令行参数指定的观测键
    )
    
    # 获取动作维度
    action_dim = dataset.replay_buffer['action'][0].shape[-1]
    obs_dim = dataset.replay_buffer['obs'][0].shape[-1]
    
    print(f"数据集大小: {len(dataset)}")
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 初始化wandb配置
    if use_wandb:
        try:
            wandb.init(
                project="Flow",
                entity="545242836-zju",
                config={
                    "learning_rate": 1e-4,
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "action_dim": action_dim,
                    "obs_dim": obs_dim,
                    "action_horizon": pred_horizon,
                    "architecture": "Flow-Matching",
                    "optimizer": "AdamW",
                    "env_id": env_id,
                    "task_name": task_name
                }
            )
        except Exception as e:
            print(f"wandb.init失败: {e}")
            print("将禁用wandb记录")
            use_wandb = False
    
    # 创建模型
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim
    ).to(device)
    
    # 创建优化器和学习率调度器
    sigma = 0.0
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75
    )
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4,
        weight_decay=1e-6
    )
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
    
    # 创建Flow Matching对象
    FM = ConditionalFlowMatcher(sigma=sigma)
    
    # 检查点保存目录
    checkpoint_dir = f"./checkpoints/{task_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练模型
    if args.train:
        print("开始训练模型...")
        train_model(
            noise_pred_net=noise_pred_net,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            num_epochs=num_epochs,
            obs_horizon=obs_horizon,
            device=device,
            FM=FM,
            ema=ema,
            checkpoint_dir=checkpoint_dir,
            use_wandb=use_wandb  # 传入wandb参数
        )
    
    # 测试模型
    if args.test:
        # 如果指定了检查点，则使用指定的检查点
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            # 否则使用最新的检查点
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            else:
                print("错误: 没有找到可用的检查点")
                return
        
        print(f"使用检查点: {checkpoint_path}")
        
        # 测试模型
        test_model(
            noise_pred_net=noise_pred_net,
            checkpoint_path=checkpoint_path,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            device=device,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            env_id=env_id,
            env_kwargs=env_kwargs,
            render=True
        )
    
    # 确保程序结束时关闭wandb
    if use_wandb and wandb.run is not None:
        try:
            wandb.finish()
        except:
            pass


if __name__ == '__main__':
    main() 