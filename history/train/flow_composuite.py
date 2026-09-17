#!/usr/bin/env python

import sys
sys.dont_write_bytecode = True
sys.path.append('./external/models')
sys.path.append('./external')

import os
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
from composuite_dataset import CompoSuiteDataset
import composuite


def train_model(noise_pred_net, optimizer, lr_scheduler, dataloader, num_epochs, obs_horizon, device, FM, ema, checkpoint_dir):
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
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录训练损失
    loss_history = []
    
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        
        for data in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            x_img = data['obs'][:, :obs_horizon].to(device)
            x_traj = data['action'].to(device)

            x_traj = x_traj.float()
            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

            # 编码观测特征
            obs_cond = x_img.flatten(start_dim=1)
            # print(f"obs_cond.shape: {obs_cond.shape}")  # (64, 93)
            vt = noise_pred_net(xt, timestep, global_cond=obs_cond)

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # 更新EMA模型权重
            ema.step(noise_pred_net.parameters())

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
                'loss': avg_loss_train.item()
            }, checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))
    plt.close()
    
    return loss_history


def test_model(noise_pred_net, checkpoint_path, obs_horizon, action_horizon, device, pred_horizon, action_dim, task_components, render=True):
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
        task_components: 任务组件(robot, object, obstacle, objective)
        render: 是否渲染环境
    """
    # 加载模型权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
    noise_pred_net.eval()
    
    # 设置MuJoCo环境变量
    os.environ['MUJOCO_GL'] = 'glfw'
    os.environ['PYOPENGL_PLATFORM'] = 'glfw'  # 与MUJOCO_GL保持一致    
    # 创建CompoSuite环境
    robot_name, object_name, obstacle_name, objective_name = task_components
    print(f"初始化环境：{robot_name}, {object_name}, {obstacle_name}, {objective_name}")
    
    # 确保创建环境时设置了has_renderer参数
    env = composuite.make(
        robot=robot_name, 
        obj=object_name, 
        obstacle=obstacle_name, 
        task=objective_name, 
        use_task_id_obs=True, 
        has_renderer=render,               # 启用渲染器
        has_offscreen_renderer=render,     # 启用离屏渲染
        controller="joint",                # 使用关节位置控制
        env_horizon=500                    # 环境步数上限
    )
    # use_task_id_obs=False，则输入环境维度为77 ； use_task_id_obs=True，则输入环境维度为93；和数据集不对应记得改
    print(f"环境对象类型: {type(env)}")
    print(f"环境底层对象类型: {type(env.env)}")
    
    # 创建视频保存目录
    os.makedirs("videos", exist_ok=True)
    
    # 测试参数
    max_steps = 500
    test_start_seed = 10000
    n_test = 3
    
    # 运行多次测试
    for test_idx in range(n_test):
        seed = test_start_seed + test_idx
        env.seed(seed)
        
        # 重置环境
        obs = env.reset()
        
        # 初始化视频帧列表
        frames = []
        if render:
            try:
                # 对于gym封装的环境，直接使用render()方法
                frame = env.env.sim.render(camera_name="agentview", height=512, width=1024)
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    frames.append(frame)
                    print(f"成功渲染初始帧，尺寸: {frame.shape}")
                else:
                    print("初始帧渲染为空或无效")
            except Exception as e:
                print(f"渲染失败，详细错误: {str(e)}")
        
        # 初始化观测队列
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        # 记录奖励和步数
        rewards = []
        done = False
        step_idx = 0
        
        # 运行测试
        with tqdm(total=max_steps, desc=f"Eval {task_components}") as pbar:
            while not done:
                # 准备观测数据
                x_img = np.stack([x for x in obs_deque])
                x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                
                # 使用模型预测动作
                with torch.no_grad():
                    # 获取观测特征
                    obs_cond = x_img.flatten(start_dim=1)
                    # print(noise_pred_net)
                    # print(f"模型期望的输入维度: {noise_pred_net.conditioner.in_features}")
                    # print(f"实际输入维度: {obs_cond.shape}") # 为什么输入环境维度变成77了
                    # 349-333=16 ； 93-77=16 巧不巧

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
                        obs, reward, done, info = env.step(actions[j])
                        
                        # 更新观测队列和奖励
                        obs_deque.append(obs)
                        print(reward)
                        rewards.append(reward)
                        
                        # 渲染环境
                        if render:
                            try:
                                # 直接从底层环境的sim对象获取渲染结果
                                frame = env.env.sim.render(camera_name="agentview", height=512, width=1024)
                                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                                    frames.append(frame)
                            except Exception as e:
                                print(f"渲染失败: {str(e)}")
                        
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
        # print(f"累计奖励: {sum(rewards)}")
        print(f"平均奖励: {sum(rewards)/len(rewards) if rewards else 0}")
        
        # 保存视频
        if render and frames:
            try:
                if len(frames) > 0:
                    print(f"帧数: {len(frames)}, 帧形状: {frames[0].shape}")
                    video_path = f"videos/test_seed_{seed}_{robot_name}_{object_name}_{objective_name}.mp4"
                    print(f"保存视频到 {video_path}")
                    vwrite(video_path, np.array(frames))
                    print("视频已保存!")
                else:
                    print("没有有效帧，无法保存视频")
            except Exception as e:
                print(f"保存视频失败: {e}")
    
    # 释放渲染资源：
    try:
        if hasattr(env, 'env') and hasattr(env.env, 'close'):
            env.env.close()
        if hasattr(env, 'close'):
            env.close()
        
        # 释放可能的OpenGL资源
        if 'EGL_DISPLAY' in globals():
            from OpenGL import EGL
            try:
                EGL.eglTerminate(EGL_DISPLAY)
            except:
                pass
    except Exception as e:
        print(f"清理资源时发生错误: {e}")
    return rewards


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='CompoSuite Flow Matching训练与评估')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--test', action='store_true', help='测试模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='加载特定检查点')
    args = parser.parse_args()
    
    # if args.train:
    #     # 设置设备
    #     torch.cuda.set_device(1)
    #     device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # else:
    #     torch.cuda.set_device(0)
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"当前使用GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
    
    # 参数设置
    task_name = "Panda_Box_GoalWall_PickPlace"  # 任务名称，用于加载数据和保存模型
    dataset_path = f"./episodes_data/{task_name}/episodes"  # 数据集路径
    
    # 模型参数
    obs_horizon = 1  # 观测序列长度
    pred_horizon = 16  # 预测序列长度
    action_horizon = 8  # 执行的动作序列长度
    
    # 训练参数
    batch_size = 64
    num_epochs = 1000
    num_workers = 4
    
    # 创建数据集
    print(f"加载数据集: {dataset_path}")
    dataset = CompoSuiteDataset(
        dataset_dir=dataset_path,
        horizon=pred_horizon,
        pad_before=0,
        pad_after=0,
        val_ratio=0.1
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
            checkpoint_dir=checkpoint_dir
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
        # checkpoint = torch.load(checkpoint_path, map_location=device)
        # print(f"检查点中模型结构: {checkpoint.keys()}")
        
        # 解析任务名称获取组件
        parts = task_name.split('_')
        if len(parts) >= 4:
            robot_name = parts[0]
            object_name = parts[1]
            obstacle_name = parts[2]
            objective_name = '_'.join(parts[3:])
        else:
            print("错误: 无法从任务名称解析组件")
            return
        
        # 测试模型
        test_model(
            noise_pred_net=noise_pred_net,
            checkpoint_path=checkpoint_path,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            device=device,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            task_components=(robot_name, object_name, obstacle_name, objective_name),
            render=True
        )


if __name__ == '__main__':
    main() 