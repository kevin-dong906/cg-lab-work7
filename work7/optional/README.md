姓名：何佳林    
学号：202411081001   
专业：计算机科学与技术（公费师范）

# 实验报告：LBS蒙皮扩展——姿态动画（选做部分）

## 一、实验目的

在必做 LBS 可视化实验的基础上，进一步完成一个简单的姿态动画，具体要求：

1. **固定 shape 参数**：保持人体体型不变，仅改变姿态。
2. **关节旋转动画**：选择某一个关节（如右肘），使其从 0 度逐渐旋转到某个目标角度（如 90 度）。
3. **生成动画序列**：输出若干帧图片，或导出为 GIF/MP4 视频。
4. **观察蒙皮效果**：观察权重区域如何随骨骼运动被平滑带动，理解 LBS 的加权混合机制。

通过该选做内容，直观感受蒙皮权重的实际作用，加深对线性混合蒙皮动态过程的理解。

---

## 二、实验原理

### 2.1 姿态动画生成流程

基于必做实验中的手写 LBS 实现，逐帧更新姿态参数 `body_pose` 中指定关节的轴角值，然后重新执行完整的 LBS 计算（形状校正、姿态校正、蒙皮），得到该姿态下的最终网格。将所有帧的网格渲染为图片，合成连续动画。

### 2.2 关节角度插值

选取一个关节（如右肘，索引 19），在其对应的 3 维轴角向量中，绕某一坐标轴（如 Z 轴）从 0 线性变化到目标角度（如 1.2 弧度，约 69 度）。每帧角度 = 起始 + (目标 - 起始) * (t / 总帧数)。

### 2.3 动画输出方式

- 使用 `matplotlib` 渲染每一帧的 3D 网格，保存为 PNG 图片。
- 使用 `imageio` 或 `PIL` 将所有图片合成为 GIF，或使用 `ffmpeg` 生成 MP4。

---

## 三、实现步骤

### 3.1 准备环境与加载模型

沿用必做实验的环境，加载 SMPL 模型，设置固定的 `betas`（如使用必做中的形状参数）。

### 3.2 定义关节动画参数

- 选择关节索引：`joint_idx = 19`（右肘，对应 SMPL 的第 20 个关节，从 0 开始计数）。
- 目标旋转角度：绕 Z 轴旋转 `target_angle = 1.2` 弧度（约 69 度），也可尝试绕 Y 轴或 X 轴。
- 总帧数：`num_frames = 30`。

### 3.3 逐帧生成姿态

对于第 `f` 帧（`f=0..num_frames-1`）：
- 计算当前角度：`angle = (f / (num_frames-1)) * target_angle`
- 构建 `body_pose` 张量（23×3），除指定关节外其余为 0。
- 使用手写 LBS 函数（或官方前向）计算最终顶点和变换后的关节。
- 渲染网格并保存为 `frame_xxx.png`。

### 3.4 合成 GIF

使用 `imageio` 库读取所有帧图片，合并为 GIF 动画。

---

## 四、关键代码实现

### 4.1 辅助函数：设置关节角度

```python
def set_joint_pose(body_pose, joint_idx, axis_angle):
    """
    body_pose: shape (1, 23*3) tensor
    joint_idx: 关节索引 (0~22, 因为全局旋转单独处理)
    axis_angle: 3维轴角向量 (x, y, z)
    """
    start = joint_idx * 3
    body_pose[0, start:start+3] = torch.tensor(axis_angle, dtype=body_pose.dtype, device=body_pose.device)
    return body_pose
```

### 4.2 动画生成主函数

```python
def generate_animation(model, betas, joint_idx, target_angle, num_frames, out_dir):
    device = betas.device
    dtype = betas.dtype
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)
    
    # 创建输出目录
    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    frames = []
    for f in range(num_frames):
        t = f / (num_frames - 1) if num_frames > 1 else 0.0
        angle = t * target_angle
        axis_angle = [0.0, 0.0, angle]  # 绕 Z 轴旋转
        
        body_pose = torch.zeros((1, 23*3), dtype=dtype, device=device)
        body_pose = set_joint_pose(body_pose, joint_idx, axis_angle)
        
        # 使用必做中的 compute_manual_lbs 计算最终顶点
        data = compute_manual_lbs(model, betas, global_orient, body_pose)
        verts = data["verts"][0].detach().cpu().numpy()
        J_transformed = data["J_transformed"][0].detach().cpu().numpy()
        
        # 渲染
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(111, projection="3d")
        # 使用必做中的 draw_mesh 函数
        draw_mesh(ax, verts, model.faces, joints=J_transformed, title=f"Frame {f}: angle={angle:.2f} rad")
        plt.tight_layout()
        frame_path = os.path.join(frame_dir, f"frame_{f:03d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(frame_path)
        print(f"已生成帧 {f+1}/{num_frames}")
    
    # 合成 GIF
    import imageio
    images = [imageio.imread(fp) for fp in frames]
    gif_path = os.path.join(out_dir, "joint_animation.gif")
    imageio.mimsave(gif_path, images, duration=0.1)  # 每帧 0.1 秒
    print(f"动画已保存至: {gif_path}")
```

### 4.3 可视化辅助函数（增强）

为了更清晰地观察权重随骨骼运动的变化，可在每一帧叠加显示蒙皮权重的热力图（例如显示右肘关节的权重），这样能直观看到高权重区域如何被“带动”。

在 `draw_mesh` 中增加 `vertex_scalar` 参数，传入该关节的权重向量即可。

```python
# 在循环中获取该关节的权重标量
weight_scalar = model.lbs_weights[:, joint_idx].detach().cpu().numpy()
# 在绘制时传入
draw_mesh(ax, verts, faces, joints=J_transformed, vertex_scalar=weight_scalar, title=...)
```

---

## 五、运行结果与观察

### 5.1 参数设置

| 参数 | 值 |
| :--- | :--- |
| 选择关节 | 右肘 (joint_idx=19) |
| 旋转轴 | Z 轴（对应肘关节屈曲） |
| 起始角度 | 0 rad |
| 终止角度 | 1.2 rad（约 69 度） |
| 总帧数 | 30 |
| 形状参数 | 与必做一致（[2.0, -1.2, 0.8, 0, ...]） |

### 5.2 动画过程观察

- **初始帧**：人体处于 T-pose，右臂自然下垂，肘关节权重区域（颜色）主要分布在前臂。
- **中间帧**：右肘逐渐弯曲，前臂绕肘关节旋转，蒙皮权重区域平滑地跟随移动，肘部内侧出现褶皱（姿态校正补偿）。
- **最终帧**：右臂弯曲约 69 度，前臂与上臂形成夹角，肘部皮肤凸起，权重区域明显向前臂方向延伸，且边缘过渡自然，无断裂。

### 5.3 权重区域随骨骼运动的平滑带动

通过热力图叠加观察，可见：
- 肘关节权重高的顶点（红色）主要集中在前臂和肘部，随着骨骼旋转，这些顶点平滑地随前臂移动。
- 权重过渡区域（橙黄至蓝色）均匀变化，未出现突变，证明了 LBS 加权混合的有效性。
- 姿态校正（`pose_offsets`）在肘部弯曲区域增加了局部凸起，使蒙皮更真实。

---

## 六、结果文件与展示

- 输出目录结构：
  ```
  outputs/
  ├── frames/
  │   ├── frame_000.png
  │   ├── frame_001.png
  │   └── ...
  └── joint_animation.gif
  ```
- 可通过 `joint_animation.gif` 直接查看动画效果。

---

## 七、思考与讨论

1. **权重区域的响应速度**：由于权重是静态的，顶点位置完全由加权变换决定，因此运动响应是即时的，无延迟。

2. **加权混合的优势**：在肘关节弯曲时，前臂和上臂之间的顶点权重由两个关节共同控制，实现了自然弯曲，避免了“纸板式”折断。

3. **姿态校正的贡献**：在弯曲过程中，`pose_offsets` 在肘部内侧产生凸起，外侧产生拉伸，使皮肤变形更真实，这解释了为什么 LBS 前需要姿态校正。

4. **潜在问题**：大角度旋转时，若权重分布不合理，可能出现“糖果包装纸”效应（表面扭曲），本例中因 SMPL 权重经过学习，表现良好。

---

## 八、总结

本次选做通过生成单关节旋转动画，直观验证了 LBS 蒙皮在动态下的平滑效果。主要收获包括：

- 掌握了姿态参数逐帧变化与网格重计算的循环逻辑。
- 观察了蒙皮权重区域随骨骼运动平滑变形的过程，加深了对加权混合机制的理解。
- 体验了动画生成与 GIF 合成的完整流程，为后续制作更复杂的人体动画打下基础。

---

## 九、参考资料

- 必做实验代码与文档
- `smplx` 官方文档
- `imageio` 与 `matplotlib` 官方示例
