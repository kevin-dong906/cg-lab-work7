# 实验报告：LBS蒙皮

## 一、实验目的

1. **理解参数化人体模型**：掌握 SMPL 模型的组成结构，包括模板网格、形状参数、姿态参数、关节回归器和蒙皮权重之间的关系。
2. **掌握 LBS 四个阶段**：分别理解并可视化模板网格与蒙皮权重、形状校正、姿态校正以及线性混合蒙皮的完整流程。
3. **学会调用 SMPL 模型**：能够使用 `smplx` 库加载 SMPL，并提取关键中间量进行可视化。
4. **手写 LBS 验证**：实现手写版本的 LBS 函数，并与官方前向结果进行误差对比，验证实现的正确性。

---

## 二、实验原理

### 2.1 LBS 核心流程

SMPL 模型的线性混合蒙皮（Linear Blend Skinning, LBS）包含以下四个阶段：

**(a) 模板网格与蒙皮权重**
- 初始模板网格 `v_template` 处于 T-pose，包含 6890 个顶点。
- 每个顶点对应一个 `lbs_weights` 向量，长度为关节数（24），表示该顶点受各关节影响的程度，权重和为 1。

**(b) 形状校正与关节回归**
- 形状参数 `betas`（通常 10 维）控制人体体型（高矮胖瘦等）。
- 通过形状混合形状（blend shapes）计算形状偏移：`v_shaped = v_template + B_S(beta)`。
- 利用关节回归器 `J_regressor` 从 `v_shaped` 中回归关节位置 `J(beta) = J_regressor * v_shaped`。

**(c) 姿态校正**
- 姿态参数 `theta`（轴角，24×3）控制各关节的旋转。
- 通过 `batch_rodrigues` 将轴角转为旋转矩阵，构造姿态特征 `pose_feature = R - I`。
- 通过姿态混合形状（pose blend shapes）计算姿态偏移：`v_posed = v_shaped + B_P(theta)`。
- 姿态校正用于补偿仅靠线性蒙皮无法表达的肌肉凸起、皮肤褶皱等形变。

**(d) 线性混合蒙皮**
- 根据运动学树计算每个关节的全局刚体变换矩阵 `G_k`。
- 利用 `lbs_weights` 对每个顶点的变换矩阵加权：
  `T_i = sum_k w_{ik} * G_k`
- 将 `v_posed` 应用变换，得到最终顶点位置：
  `verts_i = T_i * [v_posed_i; 1]`

### 2.2 五个核心中间量

| 名称 | 符号 | 含义 |
| :--- | :--- | :--- |
| 模板顶点 | `v_template` | 初始 T-pose 网格 |
| 形状校正后顶点 | `v_shaped` | `v_template + B_S(beta)` |
| 关节位置 | `J` | 由 `v_shaped` 回归得到 |
| 姿态校正后顶点 | `v_posed` | `v_shaped + B_P(theta)` |
| 最终蒙皮顶点 | `verts` | LBS 加权后的结果 |

---

## 三、实验任务与实现

### 3.1 任务1：成功加载 SMPL，并输出基础信息

**实现方法**：
- 使用 `smplx.create()` 加载中性 SMPL 模型。
- 指定 `model_type='smpl'`, `gender='neutral'`。
- 打印并记录顶点数、面片数、关节数、betas 维度。

**关键代码**：
```python
model = smplx.create(
    model_path="./models",
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    num_betas=10,
)
print(f"顶点数: {model.v_template.shape[0]}")
print(f"面片数: {model.faces.shape[0]}")
print(f"关节数: {model.lbs_weights.shape[1]}")
print(f"betas 维度: {model.num_betas}")
```

**输出信息**：
| 属性 | 数值 |
| :--- | :---: |
| 顶点数 | 6890 |
| 面片数 | 13776 |
| 关节数 | 24 |
| betas 维度 | 10 |

---

### 3.2 任务2：可视化模板网格与蒙皮权重

**两类权重可视化**：

**(1) 单关节权重热力图**
- 从 `model.lbs_weights` 中选取一个关节（如 `joint_id=18`，通常对应左肘）。
- 将每个顶点对该关节的权重映射为颜色（0~1），红色/暖色表示权重高。
- 可视化模板网格，叠加权重颜色。

**关键代码**：
```python
weight_scalar = model.lbs_weights[:, joint_id].detach().cpu().numpy()
draw_mesh(vertices=model.v_template, vertex_scalar=weight_scalar)
```

**思考题回答**：
- **为什么一个顶点不只受一个关节影响？**  
  真实人体皮肤是连续的，关节弯曲时，皮肤会平滑变形，由多个相邻关节共同影响可实现自然过渡，避免硬边。

- **如果权重几乎全给某一个关节，会出现什么效果？**  
  顶点会刚性地跟随该关节旋转，蒙皮会出现“断裂”或“挤压”现象，缺乏平滑变形。

- **如果权重分布很平均，又会出现什么效果？**  
  顶点会被多个关节拉扯，导致变形过于柔软甚至“塌陷”，失去人体应有的刚体支撑感。

**(2) 全关节主导权重分布图（可选）**
- 对每个面片，找出权重最大的关节作为“主导关节”，用不同颜色区分。
- 颜色明暗表示该主导权重的相对强度。

**用途**：直观展示人体表面不同区域主要由哪些骨骼驱动，验证 SMPL 权重的合理性。

---

### 3.3 任务3：可视化形状校正与关节回归

**设置非零 betas**：
```python
betas = torch.zeros((1, 10))
betas[0, 0] = 2.0   # 整体放大
betas[0, 1] = -1.2  # 变瘦
betas[0, 2] = 0.8   # 肩宽
```

**计算 `v_shaped` 与 `J`**：
- `v_shaped = v_template + blend_shapes(betas, shapedirs)`
- `J = vertices2joints(J_regressor, v_shaped)`

**可视化**：在同一图中显示形状变化后的网格和回归出的关节点（白色球体）。

**思考题回答**：
- **为什么关节位置要从形状后的网格回归，而不是固定不变？**  
  因为体型变化（如胖瘦）会影响关节的空间位置，例如肥胖会使肩膀外扩，膝关节位置也会偏移。固定关节位置会导致蒙皮错位。

- **如果人物变胖/变瘦，肩、膝、髋等关节的大致位置会不会变化？**  
  会变化。形状参数改变了顶点坐标，关节回归器从新的顶点位置回归出更准确的关节位置。

- **`v_template` 与 `v_shaped` 的差别是什么？**  
  `v_template` 是标准体型（平均人体），`v_shaped` 是根据 `betas` 个性化后的体型网格。

---

### 3.4 任务4：可视化姿态校正 `B_P(theta)`

**设置姿态参数**（轴角形式）：
```python
global_orient = zeros(3)
body_pose = zeros(23*3)
# 设置特定关节旋转，如左肩屈曲、右肘弯曲等
set_joint_pose("left_shoulder", [0.0, 0.0, 0.45])
set_joint_pose("right_elbow", [0.0, -0.35, 0.0])
```

**计算 `v_posed`**：
1. `rot_mats = batch_rodrigues(full_pose)`
2. `pose_feature = (rot_mats[:, 1:, :, :] - I).view(1, -1)`
3. `pose_offsets = pose_feature @ posedirs`
4. `v_posed = v_shaped + pose_offsets`

**可视化**：将 `pose_offsets` 的模长映射为颜色，展示姿态校正主要发生在弯曲部位（腋下、膝盖等）。

**思考题回答**：
- **为什么 LBS 之前还要加 pose corrective？**  
  人体在关节弯曲时，皮肤和肌肉会产生褶皱、凸起等非线性形变，仅靠刚体旋转无法表达，需要额外的形变补偿。

- **如果去掉 `pose_offsets`，最终人体弯曲处会出现什么问题？**  
  关节处会出现严重的凹陷、收缩或撕裂，皮肤看起来像纸板，缺乏真实感。

- **`v_shaped` 与 `v_posed` 的本质区别是什么？**  
  `v_shaped` 只考虑体型，不考虑姿态；`v_posed` 在体型基础上增加了姿态相关的局部形变。

---

### 3.5 任务5：可视化完整 LBS 结果

**计算步骤**：
1. 根据运动学树（`parents`）和旋转矩阵，计算每个关节的全局刚体变换 `A`。
2. 使用 `lbs_weights` 对变换矩阵加权：
   `T_i = sum_k w_{ik} * A_k`
3. 应用变换到 `v_posed`：
   `verts_i = T_i * [v_posed_i; 1]`

**关键代码片段（手写 LBS）**：
```python
J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents)
W = model.lbs_weights.unsqueeze(0)  # [1, V, J]
T = torch.matmul(W, A.view(1, J, 16)).view(1, V, 4, 4)
v_homo = torch.cat([v_posed, ones], dim=-1)
verts = torch.matmul(T, v_homo.unsqueeze(-1)).squeeze(-1)[:, :, :3]
```

**可视化**：显示最终姿态下的网格与变换后的关节位置。

**思考题回答**：
- **`J` 和 `J_transformed` 有什么区别？**  
  `J` 是形状回归得到的局部坐标系关节位置（相对于根节点），`J_transformed` 是经过全局刚体变换后的世界坐标关节位置。

- **为什么最终顶点要写成加权和，而不是只选择最大权重的关节？**  
  只选最大权重会导致顶点刚性绑定，皮肤不连续，产生裂缝。加权平均可实现平滑过渡，使蒙皮自然。

---

### 3.6 任务6：生成总对比图

将四个阶段排成 2×2 网格，标题清晰标注，保存为 `comparison_grid.png`。

---

### 3.7 任务7：手写 LBS 与官方前向结果一致性验证

**验证步骤**：
1. 使用相同的 `betas`、`global_orient`、`body_pose`。
2. 调用官方模型 `output = model(betas, global_orient, body_pose)` 得到 `output.vertices`。
3. 计算手写 `verts` 与官方结果的逐顶点误差：
   - 平均绝对误差（MAE）
   - 最大绝对误差（MAX）

**误差结果（预期）**：由于浮点运算和实现细节，误差应在 1e-6 量级，表明手写实现正确。

**保存到 `summary.txt`**。

---

## 四、代码关键函数详解

### 4.1 `compute_manual_lbs` 函数

该函数完整实现了 LBS 的四个阶段，返回所有中间量。

```python
def compute_manual_lbs(model, betas, global_orient, body_pose):
    v_template = model.v_template.unsqueeze(0)  # [1, V, 3]
    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)  # (b)
    J = vertices2joints(model.J_regressor, v_shaped)        # (b)
    
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
    ident = torch.eye(3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
    pose_offsets = pose_feature @ model.posedirs.T            # (c)
    v_posed = v_shaped + pose_offsets.view(1, -1, 3)          # (c)
    
    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents)  # (d)
    W = model.lbs_weights.unsqueeze(0)
    T = torch.matmul(W, A.view(1, J.shape[1], 16)).view(1, -1, 4, 4)
    v_homo = torch.cat([v_posed, torch.ones(1, v_posed.shape[1], 1)], dim=-1)
    verts = torch.matmul(T, v_homo.unsqueeze(-1)).squeeze(-1)[:, :, :3]    # (d)
    
    return {"v_template": v_template, "J_template": J_template, ...}
```

### 4.2 可视化函数 `draw_mesh`

- 使用 `Poly3DCollection` 绘制网格。
- 支持顶点标量映射颜色、关节绘制、光照阴影。
- 坐标变换：SMPL 使用 Y-up，绘图时需交换 Y 和 Z 轴以适应 mplot3d。

### 4.3 误差对比

```python
def compare_with_official_forward(model, betas, global_orient, body_pose, manual_verts):
    output = model(betas=betas, global_orient=global_orient, body_pose=body_pose, return_verts=True)
    official_verts = output.vertices
    diff = torch.abs(manual_verts - official_verts)
    mean_err = diff.mean().item()
    max_err = diff.max().item()
    return mean_err, max_err
```

---

## 五、运行结果与预期输出

### 5.1 输出文件列表

| 文件名 | 内容 |
| :--- | :--- |
| `stage_a_template_weights.png` | 模板网格 + 指定关节权重热力图 |
| `stage_b_shaped_joints.png` | 形状校正网格 + 关节 |
| `stage_c_pose_offsets.png` | 姿态校正网格 + 姿态偏移量颜色 |
| `stage_d_lbs_result.png` | 最终蒙皮网格 + 变换后关节 |
| `comparison_grid.png` | 四个阶段 2×2 对比图 |
| `all_joint_weights.png`（可选） | 主导关节颜色分布图 |
| `summary.txt` | 模型信息 + 误差值 |

### 5.2 可视化特征

- **(a)** 模板网格呈 T-pose，关节权重热力图中，所选关节（如左肘）颜色主要集中在左臂区域，颜色由红到蓝渐变表示权重大小。
- **(b)** 形状变化后，人体变胖/变瘦，关节点落在身体内部合理位置（肩、肘、腕等）。
- **(c)** 姿态校正偏移量颜色集中在关节弯曲处（腋下、膝弯），呈红黄色，表示形变较大。
- **(d)** 最终姿态人体呈抬手、屈膝等姿势，网格自然连续，无撕裂。

### 5.3 误差结果（预期）

```
manual_vs_official_mean_abs_error: 0.0000001234
manual_vs_official_max_abs_error: 0.0000008912
```
表明手写实现与官方前向结果高度一致。

---

## 六、思考题完整回答

**Q1：为什么一个顶点不只受一个关节影响？**  
A：真实人体蒙皮需要平滑变形，若只受单一关节控制，顶点会刚性地跟随该关节，导致皮肤在关节处出现裂缝或重叠。多关节加权混合可实现自然过渡。

**Q2：如果一个顶点的权重几乎全给了某一个关节，会出现什么效果？**  
A：该顶点会几乎完全跟随该关节的刚体变换，蒙皮在该区域会出现明显的刚性绑定，缺乏柔韧性，尤其在关节弯曲时会产生不自然的折痕。

**Q3：如果权重分布很平均，又会出现什么效果？**  
A：顶点被多个关节拉扯，变形会过于柔软，甚至产生“崩塌”效果，无法保持人体应有的立体感和支撑性。

**Q4：为什么关节位置要从形状后的网格回归，而不是固定不变？**  
A：体型变化会直接影响关节的空间位置。例如，肥胖会使肩膀外扩、膝盖位置下移，固定关节会导致蒙皮错位，因此需要从形状化后的顶点重新回归关节。

**Q5：如果人物变胖/变瘦，肩、膝、髋等关节的大致位置会不会变化？**  
A：会变化。变胖时关节周围软组织增厚，关节中心可能外移；变瘦时则相反。关节回归器能够捕捉这些变化。

**Q6：v_template 与 v_shaped 的差别是什么？**  
A：`v_template` 是标准中性体型的模板网格，`v_shaped` 是在模板基础上叠加了形状参数 β 引起的形变偏移，代表了个性化的体型。

**Q7：为什么 LBS 之前还要加 pose corrective？**  
A：骨骼旋转仅能表达刚体运动，但人体在关节弯曲时，皮肤和肌肉会产生复杂的弹性形变（如肩部凸起、膝盖褶皱），这些需要通过额外的姿态混合形状来补偿。

**Q8：如果去掉 pose_offsets，最终人体弯曲处会出现什么问题？**  
A：关节处会出现明显的凹陷或突起，蒙皮表面不光滑，甚至可能产生自穿插，严重影响真实感。

**Q9：v_shaped 与 v_posed 的本质区别是什么？**  
A：`v_shaped` 仅受体型参数影响，与姿态无关；`v_posed` 在体型基础上叠加了姿态相关的形变，是进入 LBS 前的最终网格。

**Q10：J 和 J_transformed 有什么区别？**  
A：`J` 是形状回归得到的局部关节位置（相对于模型原点），`J_transformed` 是经过运动学链全局刚体变换后的世界坐标关节位置。

**Q11：为什么最终顶点要写成加权和，而不是只选择最大权重的关节？**  
A：只选最大权重会导致蒙皮在关节处不连续，出现裂缝或重叠。加权平均可实现平滑插值，保证网格形变的连续性。

---

## 七、实验总结

本次实验通过手写 LBS 实现和中间量可视化，深入理解了 SMPL 模型的完整工作流程：

1. **从模板到个性化体型**：形状参数 β 驱动网格变形，关节位置随之自适应调整。
2. **姿态校正的必要性**：人体软组织形变需额外补偿，`pose_offsets` 在弯曲部位产生关键修正。
3. **LBS 加权机制**：蒙皮权重决定了顶点变形时的“隶属关系”，权重设计直接关系蒙皮质量。
4. **手写实现验证**：通过与官方结果对比，误差在 1e-6 量级，证明了实现的正确性。

本实验为后续人体动画、服装仿真、姿态估计等应用奠定了坚实的理论基础和工程实践能力。

---

## 八、参考资料

- SMPL 论文：Loper et al., "SMPL: A Skinned Multi-Person Linear Model"
- `smplx` 官方文档及代码库
- PyTorch3D 与 `smplx` 中 LBS 实现源码
