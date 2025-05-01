---
guided_diffusion 代码解读
---

# 一、UNet 部分

## 1. 自注意力全局池化

```python
class AttentionPool2d(nn.Module):
    """
    通过自注意力机制实现2D特征图的全局池化，输出全局特征向量
    改编自CLIP：https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(
        self,
        spacial_dim: int,         # 输入特征图的空间尺寸（假设H=W，如14表示14x14）
        embed_dim: int,           # 输入特征的通道数（C）
        num_heads_channels: int,  # 每个注意力头的通道数
        output_dim: int = None    # 可选输出维度（默认同embed_dim）
    ):
        super().__init__()
        
        # 可学习的位置编码矩阵：[C, H*W + 1]
        # 加1是因为后续会拼接一个全局平均特征
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ​**​ 2 + 1) / embed_dim ​**​ 0.5  # Xavier初始化
        )
        
        # 1x1卷积生成QKV矩阵：输入C通道，输出3*C通道（Q/K/V各C通道）
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        
        # 1x1卷积调整输出维度：输入C通道，输出output_dim或C通道
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        
        # 计算注意力头数（总通道数 / 每个头的通道数）
        self.num_heads = embed_dim // num_heads_channels
        
        # 核心注意力计算模块
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        # 输入x形状：[B, C, H, W]
        b, c, *_spatial = x.shape
        
        # 1. 展平空间维度 -> [B, C, H*W]
        x = x.reshape(b, c, -1)  # NC(HW)
        
        # 2. 在末尾拼接全局平均特征 -> [B, C, H*W + 1]
        # 这个额外特征类似CLS token，作为全局查询的锚点
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)
        
        # 3. 添加位置编码（广播到batch维度）
        x = x + self.positional_embedding[None, :, :].to(x.dtype)
        
        # 4. 生成QKV并计算注意力
        x = self.qkv_proj(x)      # -> [B, 3*C, H*W+1]
        x = self.attention(x)     # 多头注意力加权融合 -> [B, C, H*W+1]
        
        # 5. 输出投影并取第一个位置（即全局特征）
        x = self.c_proj(x)        # -> [B, output_dim, H*W+1]
        return x[:, :, 0]         # -> [B, output_dim]
```



## 2. 时间步嵌入

```python
class TimestepBlock(nn.Module):
    """
    抽象基类：所有需要结合时间步嵌入（timestep embeddings）的模块必须继承此类
    典型应用：扩散模型中UNet的残差块、注意力块等需要时间条件化的组件
    
    子类必须实现 forward(x, emb) 方法，其中：
    - x: 输入特征张量
    - emb: 时间步嵌入向量（通常通过正弦嵌入或MLP生成）
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        子类必须实现的前向传播逻辑
        Args:
            x: 输入特征，形状通常为 [B, C, ...]（具体维度由子类决定）
            emb: 时间步嵌入向量，形状通常为 [B, D_emb]
        Returns:
            经时间条件化处理后的特征，形状应与输入x一致
        """
        raise NotImplementedError
```





## 3. 时间步嵌入的顺序容器模块

```python
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    支持时间步嵌入的顺序容器模块
    继承自nn.Sequential和TimestepBlock，实现两种前向传播的自动路由：
    - 对普通层（如Conv2D）调用 layer(x)
    - 对时间条件化层（TimestepBlock子类）调用 layer(x, emb)

    典型应用场景：构建扩散模型的UNet中的残差块序列，混合普通卷积和时间条件化操作
    """

    def forward(self, x, emb):
        """
        重写的前向传播方法，自动处理时间步嵌入的分发
        Args:
            x: 输入特征张量，形状由具体层决定
            emb: 时间步嵌入向量，形状通常为 [B, D_emb]
        Returns:
            处理后的特征张量
        """
        for layer in self:
            # 类型检查实现智能路由
            if isinstance(layer, TimestepBlock):
                # 时间条件化层：传入时间和特征
                x = layer(x, emb)  # 调用TimestepBlock子类的forward(x,emb)
            else:
                # 普通层：仅传入特征
                x = layer(x)  # 调用原生nn.Module的forward(x)
        return x
```



## 4. 上采样模块

```python
class Upsample(nn.Module):
    """
    可配置卷积的上采样层，支持1D/2D/3D数据
    核心功能：先进行最近邻上采样，可选是否添加卷积层
    
    典型应用：扩散模型UNet的解码器部分，特征图上采样
    
    :param channels: 输入特征通道数
    :param use_conv: 是否在上采样后添加3x3卷积
    :param dims: 数据维度 (1/2/3对应1D/2D/3D)
    :param out_channels: 输出通道数（默认同输入）
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels  # 默认输出通道=输入通道
        self.use_conv = use_conv
        self.dims = dims
        
        # 当启用卷积时，初始化3x3卷积核（自动适应1D/2D/3D）
        if use_conv:
            self.conv = conv_nd(  # 来自utils的维度自适应卷积
                dims, 
                self.channels, 
                self.out_channels, 
                3,  # kernel_size
                padding=1  # 保持空间尺寸
            )

    def forward(self, x):
        # 输入通道数校验
        assert x.shape[1] == self.channels, \
            f"Expected {self.channels} channels but got {x.shape[1]}"
        
        # 维度特定的上采样处理
        if self.dims == 3:
            # 3D特殊处理：仅在内层两个维度上采样（DHW -> D, H*2, W*2）
            x = F.interpolate(
                x, 
                size=(x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest"  # 最近邻插值避免引入伪影
            )
        else:
            # 1D/2D标准处理：所有空间维度2倍上采样
            x = F.interpolate(
                x, 
                scale_factor=2,  # 统一缩放因子
                mode="nearest"
            )
        
        # 可选卷积操作（用于特征转换）
        if self.use_conv:
            x = self.conv(x)  # 3x3卷积保持尺寸
            
        return x
```



## 5. 下采样模块

```python
class Downsample(nn.Module):
    """
    可配置卷积的下采样层，支持1D/2D/3D数据
    核心功能：通过卷积或平均池化实现降采样
    
    典型应用：扩散模型UNet的编码器部分，逐步压缩特征图尺寸
    
    :param channels: 输入特征通道数
    :param use_conv: True=使用步长卷积, False=使用平均池化
    :param dims: 数据维度 (1/2/3对应1D/2D/3D)
    :param out_channels: 输出通道数（默认同输入）
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        
        # 3D数据特殊处理：仅在最后两个维度下采样（DHW -> D, H/2, W/2）
        stride = 2 if dims != 3 else (1, 2, 2)  # 3D时保持深度维度
        
        if use_conv:
            # 使用步长卷积实现下采样
            self.op = conv_nd(
                dims, 
                self.channels, 
                self.out_channels, 
                kernel_size=3, 
                stride=stride,  # 通过stride=2实现降采样
                padding=1  # 保持空间关系
            )
        else:
            # 使用平均池化时要求输入输出通道相同
            assert self.channels == self.out_channels, \
                "AvgPooling requires equal in/out channels"
            self.op = avg_pool_nd(
                dims, 
                kernel_size=stride, 
                stride=stride
            )

    def forward(self, x):
        # 输入通道数校验
        assert x.shape[1] == self.channels, \
            f"Expected {self.channels} channels but got {x.shape[1]}"
        return self.op(x)  # 统一接口调用
```



## 6. 残差模块

```python
class ResBlock(TimestepBlock):
    """
    时间条件化的残差块，支持通道变换/上下采样/梯度检查点等高级功能
    核心创新点：将时间嵌入通过scale-shift机制注入网络（类似AdaIN）
    
    :param use_scale_shift_norm: 是否使用"缩放-平移"归一化（更精细的时间控制）
    :param up/down: 集成上/下采样操作（替代单独层）
    :param use_checkpoint: 梯度检查点技术（节省显存）
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,  # 是否拆分emb为scale和shift
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        # 基础参数配置
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # 主分支预处理层（归一化+激活+卷积）
        self.in_layers = nn.Sequential(
            normalization(channels),  # 通常是GroupNorm
            nn.SiLU(),  # Swish激活
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # 上/下采样标志
        self.updown = up or down
        if up:
            # 上采样分支（共享参数）
            self.h_upd = Upsample(channels, False, dims)  # 特征上采样
            self.x_upd = Upsample(channels, False, dims)  # 捷径上采样
        elif down:
            # 下采样分支
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # 时间嵌入处理层
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(  # 关键投影层
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # 输出层（含归一化、激活、dropout和零初始化卷积）
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(  # 输出卷积初始化为零（稳定训练）
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # 捷径连接配置
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()  # 通道不变时直连
        elif use_conv:
            # 使用3x3卷积调整通道（保留更多空间信息）
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            # 默认用1x1卷积调整通道
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        # 启用梯度检查点（节省显存）
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # 上/下采样分支处理
        if self.updown:
            # 分离卷积操作（保持上采样前后的一致性）
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)  # 归一化+激活
            h = self.h_upd(h)  # 特征上/下采样
            x = self.x_upd(x)  # 捷径上/下采样
            h = in_conv(h)  # 卷积操作
        else:
            h = self.in_layers(x)  # 标准处理

        # 时间条件化注入
        emb_out = self.emb_layers(emb).type(h.dtype)
        # 调整emb维度（对齐特征图维度）
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]  # 补充缺失的维度

        # 核心创新点：scale-shift机制
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)  # 拆分嵌入
            h = out_norm(h) * (1 + scale) + shift  # 类似FiLM的调制
            h = out_rest(h)
        else:
            h = h + emb_out  # 基础加法融合
            h = self.out_layers(h)

        # 残差连接
        return self.skip_connection(x) + h
```



## 7. 注意力模块

```python
class AttentionBlock(nn.Module):
    """
    空间自注意力模块，使特征图的每个位置都能关注其他位置
    核心特点：
    - 支持多头注意力机制
    - 两种注意力计算顺序可选
    - 内置残差连接和梯度检查点

    原始实现参考TensorFlow版本，但适配了N维数据：
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66
    """
    def __init__(
        self,
        channels,                   # 输入特征通道数
        num_heads=1,                # 注意力头数
        num_head_channels=-1,       # 每个头的通道数（-1表示自动计算）
        use_checkpoint=False,       # 是否启用梯度检查点
        use_new_attention_order=False,  # 是否使用新版注意力计算顺序
    ):
        super().__init__()
        self.channels = channels
        
        # 多头注意力配置
        if num_head_channels == -1:
            self.num_heads = num_heads  # 直接指定头数
        else:
            # 根据每个头的通道数计算头数
            assert channels % num_head_channels == 0, \
                f"通道数{channels}必须能被num_head_channels{num_head_channels}整除"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        
        # 输入归一化层（通常为GroupNorm）
        self.norm = normalization(channels)
        
        # QKV生成卷积（1x1卷积实现）
        self.qkv = conv_nd(1, channels, channels * 3, 1)  # 输出通道为3*C（Q/K/V各C通道）
        
        # 选择注意力计算方式
        if use_new_attention_order:
            # 新版顺序：先拆分QKV再分头（更优的内存访问模式）
            self.attention = QKVAttention(self.num_heads)
        else:
            # 旧版顺序：先分头再拆分QKV（原始Transformer顺序）
            self.attention = QKVAttentionLegacy(self.num_heads)
        
        # 输出投影层（零初始化保证训练初期稳定性）
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        # 启用梯度检查点（节省显存）
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape  # 保留原始空间维度信息
        
        # 1. 展平空间维度 [B, C, H, W,...] -> [B, C, L] (L=H*W*...)
        x_flat = x.reshape(b, c, -1)
        
        # 2. 归一化后生成QKV
        qkv = self.qkv(self.norm(x_flat))  # [B, 3*C, L]
        
        # 3. 注意力计算（核心模块）
        h = self.attention(qkv)  # [B, C, L]
        
        # 4. 输出投影
        h = self.proj_out(h)  # [B, C, L]
        
        # 5. 残差连接并恢复原始形状
        return (x_flat + h).reshape(b, c, *spatial)  # [B, C, H, W,...]
```



## 8.  QKV 注意力

传统版本

```python
class QKVAttentionLegacy(nn.Module):
    """
    传统顺序的QKV注意力实现（先分头再拆分QKV）
    匹配原始Transformer的计算顺序，保持与早期版本的兼容性
    
    计算流程：
    1. 输入张量拆分为Q/K/V
    2. 按头数分割注意力计算
    3. 缩放点积注意力
    4. 多头结果合并

    :param n_heads: 注意力头数
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads  # 保存头数用于后续计算

    def forward(self, qkv):
        """
        前向传播过程

        :param qkv: 输入张量 [batch_size, (heads * 3 * channels), sequence_length]
                   包含拼接的Q/K/V矩阵
        :return: 注意力加权后的输出 [batch_size, (heads * channels), sequence_length]
        """
        bs, width, length = qkv.shape
        
        # 验证输入维度正确性 (必须能被3*头数整除)
        assert width % (3 * self.n_heads) == 0, \
            f"输入通道数{width}必须能被3*头数{3*self.n_heads}整除"
        
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        
        # 1. 重排列并拆分Q/K/V
        # [B, H*3*C, L] -> [B*H, 3*C, L] -> 拆分为3个[B*H, C, L]
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        
        # 2. 缩放因子（稳定训练的关键）
        scale = 1 / math.sqrt(math.sqrt(ch))  # 双重平方根更稳定
        
        # 3. 缩放点积注意力计算
        # 计算注意力权重 [B*H, L, L]
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        
        # 4. 注意力归一化（混合精度处理）
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        
        # 5. 注意力加权求和 [B*H, C, L]
        a = th.einsum("bts,bcs->bct", weight, v)
        
        # 6. 合并多头结果 [B, H*C, L]
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        """
        计算FLOPs的hook方法
        
        :param model: 模块实例
        :param _x: 输入张量
        :param y: 输出张量
        :return: 总浮点运算次数
        """
        return count_flops_attn(model, _x, y)  # 实际计算需实现该函数
```

改进版

```python
class QKVAttention(nn.Module):
    """
    改进版QKV注意力模块，采用更优的内存访问顺序
    核心创新：先拆分QKV再分头，提升计算效率
    
    :param n_heads: 注意力头数
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads  # 保存注意力头数

    def forward(self, qkv):
        """
        前向传播过程（优化版计算流程）

        :param qkv: 输入张量 [batch_size, (3 * heads * channels), sequence_length]
                   包含连续存储的Q/K/V矩阵
        :return: 注意力加权输出 [batch_size, (heads * channels), sequence_length]
        """
        bs, width, length = qkv.shape
        
        # 验证输入维度有效性
        assert width % (3 * self.n_heads) == 0, \
            f"输入通道数{width}必须能被3*头数{3*self.n_heads}整除"
        
        # 计算每个头的通道数
        ch = width // (3 * self.n_heads)
        
        # 1. 主拆分点：先整体拆解Q/K/V [B,3*H*C,L] -> 3x [B,H*C,L]
        q, k, v = qkv.chunk(3, dim=1)  # 相比Legacy版本，这是第一个关键差异点
        
        # 2. 缩放因子（双重平方根增强稳定性）
        scale = 1 / math.sqrt(math.sqrt(ch))  # 1/(d_k)^(1/4)
        
        # 3. 重排列维度并计算注意力权重
        # [B,H*C,L] -> [B*H,C,L] 的优化视图变换
        weight = th.einsum(
            "bct,bcs->bts",  # 爱因斯坦求和约定
            (q * scale).view(bs * self.n_heads, ch, length),  # Q矩阵
            (k * scale).view(bs * self.n_heads, ch, length)   # K矩阵
        )  # 输出形状 [B*H, L, L]
        
        # 4. 混合精度softmax（FP32计算后转回原精度）
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        
        # 5. 注意力加权求和
        a = th.einsum(
            "bts,bcs->bct",  # 爱因斯坦求和
            weight,  # 注意力权重 [B*H, L, L]
            v.reshape(bs * self.n_heads, ch, length)  # V矩阵 [B*H, C, L]
        )  # 输出形状 [B*H, C, L]
        
        # 6. 合并多头结果 [B, H*C, L]
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        """FLOPs计算接口（需配合外部函数实现）"""
        return count_flops_attn(model, _x, y)
```



## 9. UNet 模型



```python
class UNetModel(nn.Module):
    """
    完整的UNet模型，集成时间步嵌入和注意力机制
    核心功能：
    - 多尺度残差块架构
    - 可配置的注意力层位置
    - 时间步和类别条件注入
    - 内存优化设计（梯度检查点、混合精度）

    原始论文参考：https://arxiv.org/abs/2006.11239
    """

    def __init__(
        self,
        image_size,                # 输入图像尺寸（正方形）
        in_channels,               # 输入通道数（RGB=3）
        model_channels,            # 基础通道数（后续按channel_mult缩放）
        out_channels,              # 输出通道数（通常同in_channels）
        num_res_blocks,            # 每个分辨率级的残差块数量
        attention_resolutions,     # 需要添加注意力的下采样比例（如[4,8]）
        dropout=0,                 # Dropout概率
        channel_mult=(1, 2, 4, 8),# 各层级通道数倍增系数
        conv_resample=True,        # 是否使用可学习上/下采样
        dims=2,                    # 数据维度（2D图像=2）
        num_classes=None,          # 类别条件数（None表示无条件）
        use_checkpoint=False,      # 是否启用梯度检查点
        use_fp16=False,            # 是否使用半精度
        num_heads=1,               # 注意力头数
        num_head_channels=-1,       # 每头通道数（-1自动计算）
        use_scale_shift_norm=False,# 是否使用scale-shift归一化
        resblock_updown=False,     # 是否在采样层使用残差块
        use_new_attention_order=False,  # 是否使用新版注意力计算顺序
    ):
        super().__init__()
        
        # --- 基础配置验证 ---
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads  # 上采样默认头数

        # --- 核心参数存储 ---
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        # --- 时间步嵌入网络 ---
        time_embed_dim = model_channels * 4  # 嵌入维度=4*基础通道
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),  # 全连接升维
            nn.SiLU(),  # Swish激活
            linear(time_embed_dim, time_embed_dim),  # 二次映射
        )

        # --- 类别条件嵌入（可选）---
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # --- 输入块（第一层卷积）---
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(  # 支持时间嵌入的容器
                conv_nd(dims, in_channels, ch, 3, padding=1)  # 初始卷积
            )
        ])
        self._feature_size = ch  # 跟踪当前特征图通道数
        input_block_chans = [ch]  # 记录各层通道数

        # --- 下采样阶段构建 ---
        ds = 1  # 当前下采样倍数
        for level, mult in enumerate(channel_mult[1:]):  # 遍历各分辨率级
            # 残差块模块
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(  # 时间条件化残差块
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                
                # 在指定分辨率添加注意力
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(  # 空间注意力块
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            # 下采样层（除最后一级）
            if level != len(channel_mult) - 2:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(  # 带下采样的残差块
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            down=True,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ) if resblock_updown else Downsample(  # 或独立下采样层
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2  # 更新下采样倍数
                self._feature_size += ch

        # --- 中间块（最深层）---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(  # 残差块×2 + 注意力
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(  # 核心注意力层
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # --- 上采样阶段构建 ---
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:  # 反向遍历
            for i in range(num_res_blocks + 1):  # 每级n+1个块
                ich = input_block_chans.pop()  # 获取对应下采样级通道数
                layers = [
                    ResBlock(  # 残差块（融合跳跃连接）
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                
                # 在指定分辨率添加注意力
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                
                # 上采样层（除第一级）
                if level != 0 and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(  # 带上采样的残差块
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            up=True,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ) if resblock_updown else Upsample(  # 或独立上采样层
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2  # 更新上采样倍数
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # --- 输出层 ---
        self.out = nn.Sequential(
            normalization(ch),  # 最终归一化
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),  # 零初始化卷积
        )

    def forward(self, x, timesteps, y=None):
        """
        前向传播流程
        :param x: 输入张量 [B, C, H, W]
        :param timesteps: 时间步 [B,]
        :param y: 可选类别标签 [B,]
        :return: 输出张量 [B, C, H, W]
        """
        # 时间步嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # 类别条件融合
        if y is not None:
            emb = emb + self.label_emb(y)  # 加法融合条件信息
        
        # 下采样阶段
        hs = []  # 跳跃连接缓存
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)  # 时间条件化前向
            hs.append(h)  # 记录各层输出
        
        # 中间阶段
        h = self.middle_block(h, emb)
        
        # 上采样阶段（结合跳跃连接）
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)  # 通道维度拼接
            h = module(h, emb)
        
        # 输出层
        return self.out(h)
```



## 10. Unet 解码器

```python
class EncoderUNetModel(nn.Module):
    """
    半UNet结构的编码器模型，包含时间步嵌入和注意力机制
    核心特点：
    - 仅保留UNet的下采样路径（无上采样）
    - 提供多种全局池化策略选择
    - 支持时间条件化特征提取

    典型应用场景：
    - 扩散模型的编码器阶段
    - 图像特征提取器
    - 条件生成模型的编码分支
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),  # 通道倍增系数
        conv_resample=True,         # 是否使用可学习采样
        dims=2,                     # 数据维度
        use_checkpoint=False,       # 梯度检查点
        use_fp16=False,             # 半精度
        num_heads=1,                # 注意力头数
        num_head_channels=-1,       # 每头通道数
        use_scale_shift_norm=False, # 缩放平移归一化
        resblock_updown=False,      # 残差块集成采样
        use_new_attention_order=False,
        pool="adaptive",            # 池化策略选择
    ):
        super().__init__()
        
        # --- 基础配置验证 ---
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # --- 时间步嵌入网络 ---
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),  # Swish激活
            linear(time_embed_dim, time_embed_dim),
        )

        # --- 下采样阶段构建 ---
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, ch, 3, padding=1)  # 初始卷积
            )
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1  # 当前下采样倍数

        # 多级残差块构建
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                
                # 在指定分辨率添加注意力
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
                self._feature_size += ch
            
            # 下采样层（除最后一级）
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2  # 更新下采样倍数
                self._feature_size += ch

        # --- 中间块（最深层）---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # --- 多策略池化层 ---
        self.pool = pool
        if pool == "adaptive":
            # 自适应平均池化 → 1x1卷积 → 展平
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
                zero_module(conv_nd(dims, ch, out_channels, 1)),  # 1x1卷积
                nn.Flatten(),
            )
        elif pool == "attention":
            # 注意力池化（CLIP风格）
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(  # 空间注意力池化
                    (image_size // ds),  # 当前特征图尺寸
                    ch,
                    num_head_channels,
                    out_channels,
                ),
            )
        elif pool == "spatial":
            # 空间特征平均 → MLP
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),  # 拼接所有层特征
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            # 改进版空间池化（带归一化）
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),  # 层归一化
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"不支持的池化类型: {pool}")

    def forward(self, x, timesteps):
        """
        前向传播流程
        :param x: 输入张量 [B, C, H, W]
        :param timesteps: 时间步 [B,]
        :return: 编码特征 [B, out_channels]
        """
        # 时间步嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # 下采样特征提取
        results = []
        h = x.type(self.dtype)
        
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                # 空间池化模式：记录每层全局平均特征
                results.append(h.type(x.dtype).mean(dim=(2, 3)))  # [B, C]
        
        # 中间块处理
        h = self.middle_block(h, emb)
        
        # 多模式输出处理
        if self.pool.startswith("spatial"):
            # 拼接所有层特征均值
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)  # [B, sum_channels]
            return self.out(h)
        else:
            # 标准池化路径
            h = h.type(x.dtype)
            return self.out(h)
```



## 11. 超分辨模型

```python
class SuperResModel(UNetModel):
    """
    基于UNet的超分辨率模型，通过低分辨率图像条件化生成高分辨率图像
    核心创新点：
    - 双输入通道架构（原始噪声 + 上采样低清图）
    - 动态分辨率匹配机制
    - 保持UNet的时间条件化能力

    原始论文参考：https://arxiv.org/abs/2104.07636
    """

    def __init__(self, image_size, in_channels, *args, ​**kwargs):
        """
        初始化（继承自UNetModel）
        :param image_size: 目标高分辨率尺寸
        :param in_channels: 原始输入通道数（实际输入通道数×2）
        :param args/kwargs: UNet标准参数
        """
        super().__init__(image_size, in_channels * 2, *args, ​**kwargs)  # 关键修改：输入通道翻倍
        # 示例：当in_channels=3时，实际模型输入通道为6（3噪声+3低清图）

    def forward(self, x, timesteps, low_res=None, ​**kwargs):
        """
        前向传播流程
        :param x: 输入噪声张量 [B, C, H, W]
        :param timesteps: 扩散时间步 [B,]
        :param low_res: 低分辨率条件图像 [B, C, H_lr, W_lr]
        :return: 高分辨率输出 [B, C, H, W]
        """
        # 1. 动态分辨率匹配
        _, _, new_height, new_width = x.shape  # 获取目标HR尺寸
        upsampled = F.interpolate(
            low_res, 
            size=(new_height, new_width),
            mode="bilinear",  # 双线性上采样保持平滑度
            align_corners=False
        )
        
        # 2. 通道维度拼接条件信息
        # [B, C, H, W] + [B, C, H, W] → [B, 2C, H, W]
        x = th.cat([x, upsampled], dim=1)  
        
        # 3. 执行UNet标准流程（含时间条件化）
        return super().forward(x, timesteps, **kwargs)
```



# 二、Gaussian_diffusion 部分

## 1. 初始化

1. 初始化模型参数：`model_mean_type`, `model_var_type`, `loss_type`, `rescale_timesteps`。
2. 将输入的`betas`转换为`float64`类型，并确保其为一维数组且值在(0, 1]范围内。
3. 计算时间步数`num_timesteps`。
4. 计算`alphas`及其累积乘积`alphas_cumprod`，并生成前后时间步的累积乘积。
5. 计算扩散过程中的各项系数（如`sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`等）。
6. 计算后验分布的相关参数（如`posterior_variance`, `posterior_mean_coef1`, `posterior_mean_coef2`）

```python
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        # Initialize the instance variables.
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        # Ensure the shape and value range of betas.
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # Calculate the number of time steps.
        self.num_timesteps = int(betas.shape[0])

        # Calculate alphas and related cumulative products.
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
```



## 2. 计算q(x_t | x_0)

1. 使用`_extract_into_tensor`函数从预计算的数组中提取与时间步`t`对应的数据，并调整其形状以匹配`x_start`。
2. 计算均值`mean`：通过提取`sqrt_alphas_cumprod`并乘以`x_start`。
3. 计算方差`variance`：通过提取`1.0 - alphas_cumprod`。
4. 计算对数方差`log_variance`：通过提取`log_one_minus_alphas_cumprod`。
5. 返回均值、方差和对数方差。

```python
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        # Calculate the mean of the distribution q(x_t | x_0)
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        # Calculate the variance of the distribution q(x_t | x_0)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        # Calculate the logarithm of the variance of the distribution q(x_t | x_0)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        # Return the mean, variance, and log variance of the distribution
        return mean, variance, log_variance
```





## 3. 生成噪声

1. 如果未提供噪声 `noise`，则生成与 `x_start` 形状相同的随机高斯噪声。
2. 验证噪声形状是否与输入数据 `x_start` 的形状一致。
3. 使用 `_extract_into_tensor` 提取累积乘积平方根和累积乘积平方根的补数，并分别与 `x_start` 和 `noise` 相乘后相加，生成扩散后的数据

```python
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        # If no noise is provided, generate noise with the same shape as x_start
        if noise is None:
            noise = th.randn_like(x_start)
        # Ensure the shape of the noise matches that of x_start
        assert noise.shape == x_start.shape
        # Mix the initial data x_start with noise according to the diffusion step t, and return the diffused data
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
```



## 4. 后验计算

1. **输入验证**：确保 `x_start` 和 `x_t` 的形状相同。
2. **计算后验均值**：通过 `_extract_into_tensor` 提取系数并计算后验均值 `posterior_mean`，公式为 `coef1 * x_start + coef2 * x_t`。
3. **计算后验方差和对数方差**：分别提取 `posterior_variance` 和 `posterior_log_variance_clipped`。
4. **形状验证**：确保所有输出的批次维度一致。
5. **返回结果**：返回后验均值、方差和对数方差。

```python
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        This function calculates the mean and variance of the posterior distribution
        in a diffusion model, which is used to guide the reverse process of the diffusion.

        Parameters:
        - x_start: The starting image, before any diffusion steps.
        - x_t: The image at a specific diffusion step.
        - t: The current diffusion step.

        Returns:
        - posterior_mean: The mean of the posterior distribution.
        - posterior_variance: The variance of the posterior distribution.
        - posterior_log_variance_clipped: The clipped log variance of the posterior distribution.
        """
        # Ensure the shape of the input images are consistent
        assert x_start.shape == x_t.shape
        # Calculate the mean of the posterior distribution using predefined coefficients
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # Extract the variance of the posterior distribution at the current step
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        # Extract the clipped log variance of the posterior distribution at the current step
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        # Ensure the batch dimensions of the outputs are consistent
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        # Return the calculated mean, variance, and log variance
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
```



## 5. 计算 p 的方差和均值

1. **输入处理**：接收模型、输入张量 `x`、时间步 `t` 等参数，计算模型输出 `model_output`。
2. **方差类型判断**：根据 `self.model_var_type` 判断方差类型，分为 `LEARNED`、`LEARNED_RANGE` 和固定值（`FIXED_LARGE` 或 `FIXED_SMALL`），分别计算 `model_variance` 和 `model_log_variance`。
3. **预测初始值**：定义 `process_xstart` 函数处理预测的初始值 `x_start`，根据 `self.model_mean_type` 计算预测值 `pred_xstart` 和均值 `model_mean`。
4. **返回结果**：返回包含均值、方差、对数方差和预测初始值的字典。



```python
def p_mean_variance(
        self, 
        model,           # 用于预测噪声和方差的神经网络模型
        x,               # 当前时刻t的噪声图像，形状为[N, C, H, W]
        t,               # 时间步张量，形状为[N]
        clip_denoised=True,  # 是否将去噪结果裁剪到[-1,1]范围内
        denoised_fn=None,    # 对预测的原始图像x0进行后处理的函数
        model_kwargs=None    # 传递给模型的额外参数（如条件信息）
    ):
        """
        计算扩散过程中从时刻t到t-1的反向转移分布的参数：
        - 均值（mean）
        - 方差（variance） 
        - 对数方差（log_variance）
        - 对原始图像x0的预测（pred_xstart）
        """
        if model_kwargs is None:
            model_kwargs = {}  # 初始化额外参数

        B, C = x.shape[:2]  # 获取batch size和通道数
        assert t.shape == (B,)  # 验证时间步形状是否正确
        
        # 1. 模型前向计算（预测噪声和可能的方差）
        model_output = model(x, self._scale_timesteps(t), ​**model_kwargs)

        # 2. 处理不同类型的方差建模
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 可学习方差的情况：模型输出包含噪声预测和方差预测
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            
            # 将输出拆分为噪声预测和方差预测两部分
            model_output, model_var_values = th.split(model_output, C, dim=1)
            
            if self.model_var_type == ModelVarType.LEARNED:
                # 直接学习对数方差的情况
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)  # 转换为方差
            else:
                # 学习方差范围的情况（在最小和最大对数方差之间插值）
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                
                # 将模型输出从[-1,1]映射到[0,1]作为插值系数
                frac = (model_var_values + 1) / 2  
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            # 固定方差的情况
            model_variance, model_log_variance = {
                # 固定大方差：使用预计算的后验方差（第一项特殊处理）
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                # 固定小方差：使用裁剪后的对数方差（数值更稳定）
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            
            # 将方差参数提取为与输入x相同形状的张量
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        # 3. 计算对原始图像x0的预测（去噪结果）
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)  # 应用自定义后处理
            if clip_denoised:
                x = x.clamp(-1, 1)  # 裁剪到合理范围
            return x

        # 根据不同的参数化方式计算x0预测
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                # 从噪声预测中恢复x0
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            
            # 计算反向过程的均值（关键公式）
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError()

        # 4. 返回所有计算结果
        return {
            "mean": model_mean,          # 反向过程的均值
            "variance": model_variance,  # 反向过程的方差
            "log_variance": model_log_variance,  # 方差的对数形式
            "pred_xstart": pred_xstart,  # 对原始图像的预测
        }
```



## 6. p 采样

1. 调用 `self.p_mean_variance` 方法，获取当前时间步的均值和方差信息，存储在 `out` 中。
2. 生成与输入张量 `x` 相同形状的随机噪声 `noise`。
3. 构造 `nonzero_mask`，用于控制当 `t == 0` 时不添加噪声。
4. 如果 `cond_fn` 不为 `None`，调用 `self.condition_mean` 方法更新均值。
5. 根据均值、方差和噪声计算样本 `sample`。
6. 返回一个字典，包含样本 `sample` 和预测的初始状态 `pred_xstart`。

```python
def p_sample(
        self,
        model,           # 用于预测噪声和方差的神经网络模型
        x,               # 当前时刻t的噪声图像，形状为[N, C, H, W]
        t,               # 时间步张量，形状为[N]，0表示最后一步（完全去噪）
        clip_denoised=True,  # 是否将去噪结果裁剪到[-1,1]范围内
        denoised_fn=None,    # 对预测的原始图像x0进行后处理的函数
        cond_fn=None,        # 条件梯度函数（用于引导生成）
        model_kwargs=None    # 传递给模型的额外参数（如条件信息）
    ):
        """
        从给定时间步的模型中采样x_{t-1}（执行单步去噪过程）
        
        返回:
            - 'sample': 从模型采样的随机结果
            - 'pred_xstart': 对原始图像x0的预测
        """
        # 1. 计算高斯分布的均值和方差
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # 2. 生成随机噪声（与输入x同形状）
        noise = th.randn_like(x)
        
        # 3. 创建非零掩码（当t=0时不添加噪声）
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 形状变为[N,1,1,1,...]以匹配x的维度

        # 4. 应用条件函数（如果存在）
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        # 5. 重参数化采样（关键公式）
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        """
        采样公式解析：
        - 当t≠0时：x_{t-1} = 均值 + 标准差 * 随机噪声
        - 当t=0时：x_{t-1} = 均值 （最后一步不加噪声）
        其中：
        - 标准差 = exp(0.5 * log_variance) = sqrt(variance)
        - nonzero_mask确保最后一步不加噪声
        """

        return {
            "sample": sample,            # 采样结果（去噪后的图像）
            "pred_xstart": out["pred_xstart"]  # 原始图像预测（可用于可视化）
        }
```



## 7. 完整反向扩散

```python
def p_sample_loop(
        self,
        model,           # 用于去噪的神经网络模型
        shape,           # 生成样本的形状 (N, C, H, W)
        noise=None,      # 可选的初始噪声张量（若未提供则自动生成）
        clip_denoised=True,  # 是否将去噪结果裁剪到[-1,1]范围内
        denoised_fn=None,    # 对预测的原始图像x0进行后处理的函数
        cond_fn=None,        # 条件梯度函数（用于引导生成）
        model_kwargs=None,   # 传递给模型的额外参数（如条件信息）
        device=None,         # 指定计算设备（如'cuda'）
        progress=False,      # 是否显示进度条
    ):
        """
        从模型生成完整样本（执行完整的反向扩散过程）
        
        工作流程：
        1. 从高斯噪声开始（或使用提供的噪声）
        2. 从t=T到t=0逐步去噪
        3. 返回最终生成的样本
        
        返回:
            - 最终生成的样本张量（形状与输入shape相同）
        """
        final = None  # 存储最终结果
        
        # 通过渐进式采样循环生成样本
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample  # 每次迭代更新结果
        
        # 返回最终时间步（t=0）的样本
        return final["sample"]
```



## 8. 渐进式反向扩散



```python
def p_sample_loop_progressive(
        self,
        model,           # 用于去噪的神经网络模型
        shape,           # 生成样本的形状 (N, C, H, W)
        noise=None,      # 可选的初始噪声张量（若未提供则自动生成）
        clip_denoised=True,  # 是否将去噪结果裁剪到[-1,1]范围内
        denoised_fn=None,    # 对预测的原始图像x0进行后处理的函数
        cond_fn=None,        # 条件梯度函数（用于引导生成）
        model_kwargs=None,   # 传递给模型的额外参数（如条件信息）
        device=None,         # 指定计算设备（如'cuda'）
        progress=False,      # 是否显示进度条
    ):
        """
        渐进式生成样本并返回每个时间步的中间结果
        
        参数与p_sample_loop()相同
        返回一个生成器，每次产生包含以下内容的字典：
            - 'sample': 当前时间步的采样结果
            - 'pred_xstart': 对原始图像的预测
        
        工作流程：
        1. 初始化随机噪声或使用提供的噪声
        2. 从t=T到t=0逐步去噪
        3. 每次迭代生成当前状态
        """
        # 自动确定计算设备（如果未指定）
        if device is None:
            device = next(model.parameters()).device  # 获取模型所在的设备
            
        # 验证shape参数类型
        assert isinstance(shape, (tuple, list))
        
        # 初始化噪声图像
        if noise is not None:
            img = noise  # 使用用户提供的噪声
        else:
            # 生成标准正态分布噪声（形状与输入shape相同）
            img = th.randn(*shape, device=device)
        
        # 准备时间步序列（从最大到0）
        indices = list(range(self.num_timesteps))[::-1]  # 例如 [1000,999,...,0]

        # 进度条设置
        if progress:
            # 延迟导入以避免不必要依赖
            from tqdm.auto import tqdm
            indices = tqdm(indices)  # 包装进度条

        # 反向扩散过程主循环
        for i in indices:
            # 创建当前时间步张量（形状为[batch_size]）
            t = th.tensor([i] * shape[0], device=device)
            
            # 禁用梯度计算（节省内存）
            with th.no_grad():
                # 执行单步去噪采样
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                
                # 生成当前状态（实现生成器协议）
                yield out  # 返回字典 {'sample':..., 'pred_xstart':...}
                
                # 更新图像为当前采样结果
                img = out["sample"]
```



## 9. DDIM 采样

```python
def ddim_sample(
        self,
        model,           # 用于预测噪声的神经网络模型
        x,               # 当前时刻t的噪声图像，形状为[N, C, H, W]
        t,               # 时间步张量，形状为[N]
        clip_denoised=True,  # 是否将去噪结果裁剪到[-1,1]范围内
        denoised_fn=None,    # 对预测的原始图像x0进行后处理的函数
        cond_fn=None,        # 条件梯度函数（用于引导生成）
        model_kwargs=None,   # 传递给模型的额外参数（如条件信息）
        eta=0.0,             # DDIM随机性参数（η），控制采样随机性
                             # η=0为确定性采样，η=1为DDPM原始采样
    ):
        """
        使用DDIM方法从模型采样x_{t-1}
        
        实现原理：
        基于论文《Denoising Diffusion Implicit Models》中的公式(12)
        在保持生成质量的同时，允许跳过部分扩散步骤
        
        返回包含以下内容的字典：
            - 'sample': 采样结果
            - 'pred_xstart': 对原始图像的预测
        """
        # 1. 获取预测的均值、方差和x0预测
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        # 2. 应用条件函数（如果存在）
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 3. 从预测的x0重新推导噪声ε（兼容不同预测模式）
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        # 4. 提取α累积乘积（关键扩散参数）
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)          # ᾱ_t
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape) # ᾱ_{t-1}

        # 5. 计算DDIM方差σ（公式12的系数）
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))  # 标准差的缩放因子
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)          # 额外稳定性项
        )
        """
        σ的计算解析：
        - 当η=0时，σ=0（完全确定性采样）
        - 当η=1时，σ=√(1-ᾱ_{t-1})/√(1-ᾱ_t) * √(1-ᾱ_t/ᾱ_{t-1})
                   = √[(1-ᾱ_{t-1})/(1-ᾱ_t)] * √[1-ᾱ_t/ᾱ_{t-1}]
        """

        # 6. 生成随机噪声
        noise = th.randn_like(x)

        # 7. 计算均值预测（DDIM核心公式）
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)          # x0相关项
            + th.sqrt(1 - alpha_bar_prev - sigma ​**​ 2) * eps      # 噪声相关项
        )
        """
        均值公式解析：
        μ = √ᾱ_{t-1} * x0_pred + √(1-ᾱ_{t-1}-σ²) * ε_pred
        其中：
        - 第一项：基于预测x0的确定性路径
        - 第二项：基于预测噪声的随机性路径
        """

        # 8. 创建非零掩码（当t=0时不添加噪声）
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 形状变为[N,1,1,1,...]

        # 9. 最终采样（均值 + 条件性噪声）
        sample = mean_pred + nonzero_mask * sigma * noise

        return {
            "sample": sample,            # 采样结果
            "pred_xstart": out["pred_xstart"]  # 原始图像预测（可用于可视化）
        }
```



## 10. DDIM 反向采样

```python
def ddim_reverse_sample(
        self,
        model,           # 噪声预测模型（UNet结构）
        x,               # 当前时刻t的带噪图像 [N,C,H,W]
        t,               # 当前时间步 [N]
        clip_denoised=True,  # 是否将预测值裁剪到[-1,1]
        denoised_fn=None,    # 对x_start的后处理函数
        model_kwargs=None,   # 额外模型参数（如条件信息）
        eta=0.0,             # 必须为0（确定性逆向ODE）
    ):
        """
        使用DDIM逆向ODE过程采样x_{t+1}
        
        核心功能：
        1. 根据DDIM论文的逆向过程公式（反向ODE）
        2. 实现从x_t到x_{t+1}的确定性转换
        3. 用于图像编码或精确反转扩散过程
        
        数学原理：
        x_{t+1} = √ᾱ_{t+1} * x0_pred + √(1-ᾱ_{t+1}) * ε_pred
        其中ε_pred从当前x_t和预测x0推导得出
        
        返回：
            - 'sample': 逆向采样结果
            - 'pred_xstart': 预测的原始图像x0
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"  # 仅支持确定性路径

        # 1. 获取模型预测的分布参数（均值/方差/x0预测）
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # 2. 从x_t和预测x0反推噪声ε（关键推导）
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x  # (√(1/ᾱ_t))*x_t
            - out["pred_xstart"]  # - x0_pred
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)  # ÷√(1/ᾱ_t -1)
        """
        噪声推导公式解析：
        ε_pred = [ (√(1/ᾱ_t)*x_t) - x0_pred ] / √(1/ᾱ_t - 1)
        来源于正向过程定义：
        x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * ε
        """

        # 3. 获取下一时间步的α累积乘积
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)  # ᾱ_{t+1}

        # 4. 计算逆向过程的均值预测（DDIM逆向公式）
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)  # √ᾱ_{t+1} * x0_pred
            + th.sqrt(1 - alpha_bar_next) * eps           # √(1-ᾱ_{t+1}) * ε_pred
        )
        """
        逆向ODE公式解析：
        x_{t+1} = √ᾱ_{t+1} * x0_pred + √(1-ᾱ_{t+1}) * ε_pred
        与正向过程形式对称，但时间方向相反
        """

        return {
            "sample": mean_pred,          # 逆向采样结果（无随机噪声）
            "pred_xstart": out["pred_xstart"]  # 保持原始预测（用于一致性）
        }
```





## 11. DDIM 完整反向采样

```python
def ddim_sample_loop(
        self,
        model,           # 噪声预测模型（通常为UNet结构）
        shape,           # 生成样本的形状 [batch_size, channels, height, width]
        noise=None,      # 可选的自定义噪声张量（与shape同形状）
        clip_denoised=True,  # 是否将预测值裁剪到[-1,1]范围
        denoised_fn=None,    # 对x_start预测值的后处理函数
        cond_fn=None,        # 条件函数（用于分类器引导等）
        model_kwargs=None,   # 传递给模型的额外参数（如文本嵌入）
        device=None,         # 指定计算设备（如'cuda'）
        progress=False,      # 是否显示进度条
        eta=0.0,            # DDIM随机性参数（η=0为确定性采样）
    ):
        """
        使用DDIM方法生成完整样本（封装渐进式采样过程）
        
        核心功能：
        1. 初始化随机噪声（或使用提供的噪声）
        2. 通过DDIM采样过程逐步去噪
        3. 返回最终生成结果
        
        与p_sample_loop()的区别：
        - 使用DDIM的确定性/随机性采样公式
        - 支持通过eta参数控制采样随机性
        - 通常需要更少的采样步数（20-50步）
        
        返回值：
            最终生成的样本张量（形状与输入shape相同）
        """
        final = None  # 存储最终结果
        
        # 通过渐进式采样循环生成样本
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample  # 每次迭代更新为当前采样结果
        
        # 返回最终时间步（t=0）的样本
        return final["sample"]
```





## 12. DDIM 渐进式反向采样

```python
def ddim_sample_loop_progressive(
        self,
        model,           # 噪声预测模型（通常为UNet结构）
        shape,           # 生成样本的形状 [batch_size, channels, height, width]
        noise=None,      # 可选的自定义噪声张量（与shape同形状）
        clip_denoised=True,  # 是否将预测值裁剪到[-1,1]范围
        denoised_fn=None,    # 对x_start预测值的后处理函数
        cond_fn=None,        # 条件函数（用于分类器引导等）
        model_kwargs=None,   # 传递给模型的额外参数（如文本嵌入）
        device=None,         # 指定计算设备（如'cuda'）
        progress=False,      # 是否显示进度条
        eta=0.0,            # DDIM随机性参数（η=0为确定性采样）
    ):
        """
        使用DDIM方法渐进式生成样本，并返回每个时间步的中间结果
        
        核心功能：
        1. 初始化随机噪声（或使用提供的噪声）
        2. 通过DDIM采样过程逐步去噪
        3. 生成器模式返回每个时间步的结果
        
        与p_sample_loop_progressive()的区别：
        - 使用DDIM特有的采样公式（支持eta参数）
        - 可实现确定性采样（eta=0）或随机性采样（eta>0）
        
        返回值：
            生成器，每次迭代返回包含以下内容的字典：
            - 'sample': 当前时间步的采样结果
            - 'pred_xstart': 对原始图像的预测
        """
        # 设备自动检测（如果未指定）
        if device is None:
            device = next(model.parameters()).device  # 获取模型所在的设备
            
        # 验证shape参数类型
        assert isinstance(shape, (tuple, list))
        
        # 初始化噪声图像（如果未提供则随机生成）
        if noise is not None:
            img = noise  # 使用用户提供的噪声
        else:
            # 生成标准正态分布噪声（形状与输入shape相同）
            img = th.randn(*shape, device=device)
        
        # 准备时间步序列（从T到0倒序排列）
        indices = list(range(self.num_timesteps))[::-1]  # 例如 [1000,999,...,0]

        # 进度条设置（延迟导入避免不必要依赖）
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)  # 包装进度条

        # DDIM采样主循环
        for i in indices:
            # 创建当前时间步张量（形状为[batch_size]）
            t = th.tensor([i] * shape[0], device=device)
            
            # 禁用梯度计算（节省显存）
            with th.no_grad():
                # 执行单步DDIM采样
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                
                # 生成当前状态（实现生成器协议）
                yield out  # 返回字典 {'sample':..., 'pred_xstart':...}
                
                # 更新图像为当前采样结果
                img = out["sample"]
```



## 13. 训练损失

```python
def training_losses(
        self,
        model,           # 噪声预测模型（通常为UNet）
        x_start,         # 原始输入图像 [N, C, H, W]
        t,               # 随机采样的时间步 [N]
        model_kwargs=None,  # 模型额外参数（如条件信息）
        noise=None,      # 可选的自定义噪声（与x_start同形状）
    ):
        """
        计算单个时间步的训练损失
        
        实现功能：
        1. 根据扩散过程加噪
        2. 计算模型预测与目标的差异
        3. 返回包含各项损失的字典
        
        支持损失类型：
        - KL散度（变分下界）
        - MSE（均方误差）
        - 及其重缩放版本
        
        返回：
            字典包含：
            - 'loss': 主损失值
            - 'mse': 均方误差（如适用）
            - 'vb': 变分边界项（如适用）
        """
        # 1. 初始化参数
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)  # 生成随机噪声
            
        # 2. 前向扩散过程（根据时间步t加噪）
        x_t = self.q_sample(x_start, t, noise=noise)
        
        terms = {}  # 存储各项损失
        
        # 3. KL散度损失（变分下界）
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps  # 重缩放KL损失
                
        # 4. MSE损失（主流实现）
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # 模型前向预测
            model_output = model(x_t, self._scale_timesteps(t), ​**model_kwargs)
            
            # 处理可学习方差的情况
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                # 分割噪声预测和方差预测
                model_output, model_var_values = th.split(model_output, C, dim=1)
                
                # 冻结均值预测仅优化方差
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                
                if self.loss_type == LossType.RESCALED_MSE:
                    # 重缩放变分边界项（与原始实现保持一致）
                    terms["vb"] *= self.num_timesteps / 1000.0
            
            # 确定目标值（根据参数化方式）
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],  # 预测前一时刻的图像
                ModelMeanType.START_X: x_start,        # 直接预测原始图像
                ModelMeanType.EPSILON: noise,           # 预测噪声（主流方法）
            }[self.model_mean_type]
            
            # 计算MSE损失
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ​**​ 2)  # 空间维度取平均
            
            # 组合损失项
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]  # MSE + 变分边界
            else:
                terms["loss"] = terms["mse"]  # 仅MSE损失
        else:
            raise NotImplementedError(self.loss_type)
            
        return terms
```



# 三、创建模型和使用扩散

## 1. 默认配置参数

### 1.1 扩散模型默认超参数

```python
def diffusion_defaults():
    return dict(
        learn_sigma=False,           # 是否学习噪声方差（σ）
        diffusion_steps=1000,        # 扩散步数（T=1000）
        noise_schedule="linear",     # 噪声调度方式（线性/余弦等）
        timestep_respacing="",       # 时间步重采样（如"100"表示压缩到100步）
        use_kl=False,                # 是否使用KL散度损失（用于变分扩散）
        predict_xstart=False,        # 是否直接预测原始图像x₀（否则预测噪声ε）
        rescale_timesteps=False,     # 是否重新缩放时间步（如归一化到[0,1]）
        rescale_learned_sigmas=False,# 是否重新缩放学习的σ
    )
```

### 1.2 分类器默认架构参数

```python
def classifier_defaults():
    return dict(
        image_size=64,                          # 输入图像大小
        classifier_use_fp16=False,              # 是否使用FP16加速
        classifier_width=128,                   # 网络宽度（通道数基数）
        classifier_depth=2,                     # 网络深度（残差块层数）
        classifier_attention_resolutions="32,16,8",  # 在哪些分辨率下使用注意力层
        classifier_use_scale_shift_norm=True,   # 是否在残差块中使用Scale-Shift Norm
        classifier_resblock_updown=True,        # 是否在上下采样中使用残差块
        classifier_pool="attention",             # 池化方式（"attention"或"avg"）
    )
```



### 1.3 合并扩散模型架构和扩散训练参数

```python
def model_and_diffusion_defaults():
    res = dict(
        image_size=64,                  # 图像大小
        num_channels=128,               # 初始通道数
        num_res_blocks=2,               # 每个分辨率的残差块数量
        num_heads=4,                    # 注意力头的数量
        num_heads_upsample=-1,          # 上采样时的注意力头数（-1表示同num_heads）
        num_head_channels=-1,           # 每个头的通道数（-1表示自动计算）
        attention_resolutions="16,8",   # 使用注意力的分辨率（逗号分隔）
        channel_mult="",                # 通道数乘数（如"1,2,4,8"）
        dropout=0.0,                    # Dropout率
        class_cond=False,               # 是否使用类别条件
        use_checkpoint=False,           # 是否使用梯度检查点（节省显存）
        use_scale_shift_norm=True,      # 是否使用Scale-Shift Norm
        resblock_updown=False,          # 是否在上下采样中使用残差块
        use_fp16=False,                 # 是否使用FP16
        use_new_attention_order=False,   # 是否使用新的注意力顺序（优化内存）
    )
    res.update(diffusion_defaults())  # 合并扩散训练参数
    return res
```



### 1.4 合并**分类器参数**和**扩散训练参数**

```python
def classifier_and_diffusion_defaults():
    res = classifier_defaults()      # 分类器参数
    res.update(diffusion_defaults())  # 扩散训练参数
    return res
```



### 1.5 合并超分辨模型和扩散训练参数

```python
def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res
```



## 2. 创建扩散模型

### 2.1 **联合创建**UNet模型**和**高斯扩散过程

`create_model_and_diffusion()`

- **输入**：接收扩散模型和UNet的全部超参数（如图像尺寸、通道数、噪声调度等）。
- **输出**：返回配置好的`UNet模型`和`高斯扩散过程`对象。



### 2.2 构建基于UNet的扩散模型主干网络

`create_model()`

- **多尺度特征提取**：通过`channel_mult`控制不同分辨率的通道数。
- **注意力机制**：在指定分辨率（`attention_resolutions`）插入注意力层。
- **条件生成**：通过`class_cond`支持类别条件输入。



## 3. 创建超分辨扩散模型

### 3.1 **联合创建**超分辨模型**和**高斯扩散过程

`sr_create_model_and_diffusion`

### 3.2 构建基于超分辨模型的扩散模型主干网络

`sr_create_model`



# 四、重建扩散 DIRE

```python
def main():
    # 解析命令行参数
    args = create_argparser().parse_args()

    # 设置分布式环境（基于CUDA_VISIBLE_DEVICES）
    dist_util.setup_dist(os.environ["CUDA_VISIBLE_DEVICES"])
    # 配置日志记录器，输出到指定重建目录
    logger.configure(dir=args.recons_dir)

    # 创建输出目录（如果不存在）
    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)
    logger.log(str(args))  # 记录所有参数

    # 创建模型和扩散模型
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # 加载预训练模型权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())  # 将模型移动到指定设备（GPU/CPU）
    logger.log("have created model and diffusion")
    
    # 如果启用FP16，将模型转换为半精度
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()  # 设置为评估模式

    # 加载图像数据
    data = load_data_for_reverse(
        data_dir=args.images_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond
    )
    logger.log("have created data loader")

    logger.log("computing recons & DIRE ...")
    have_finished_images = 0  # 已完成图像计数器

    # 主处理循环，直到完成指定数量的样本
    while have_finished_images < args.num_samples:
        # 动态调整批次大小以精确达到目标样本数
        if (have_finished_images + MPI.COMM_WORLD.size * args.batch_size) > args.num_samples and (
            args.num_samples - have_finished_images
        ) % MPI.COMM_WORLD.size == 0:
            batch_size = (args.num_samples - have_finished_images) // MPI.COMM_WORLD.size
        else:
            batch_size = args.batch_size

        all_images = []  # 存储所有图像
        all_labels = []  # 存储所有标签（如果启用类别条件）

        # 获取下一批数据
        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]  # 裁剪到当前批次大小
        paths = paths[:batch_size]

        # 将图像数据移动到计算设备
        imgs = imgs.to(dist_util.dev())
        model_kwargs = {}
        
        # 如果启用类别条件，生成随机类别标签
        if args.class_cond:
            classes = th.randint(
                low=0,
                high=NUM_CLASSES,
                size=(batch_size,),
                device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        # 使用DDIM反向采样获取潜在表示
        reverse_fn = diffusion.ddim_reverse_sample_loop
        imgs = reshape_image(imgs, args.image_size)  # 调整图像尺寸

        # 执行反向扩散过程
        latent = reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        # 选择采样函数（DDPM或DDIM）
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        
        # 执行前向扩散过程生成重建图像
        recons = sample_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        # 计算DIRE（Deep Image Reconstruction Error）
        dire = th.abs(imgs - recons)

        # 后处理：将图像数据从[-1,1]范围转换到[0,255]的uint8
        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1)  # 从NCHW转为NHWC格式
        recons = recons.contiguous()

        # 原始图像相同处理
        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.contiguous()

        # DIRE图像处理（缩放并转换为uint8）
        dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        dire = dire.permute(0, 2, 3, 1)
        dire = dire.contiguous()

        # 分布式环境下的数据收集
        gathered_samples = [th.zeros_like(recons) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, recons)  # 收集所有进程的重建结果

        # 将结果转换为numpy数组
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        have_finished_images += len(all_images) * batch_size  # 更新已完成计数

        # 保存结果到文件
        recons = recons.cpu().numpy()
        for i in range(len(recons)):
            # 处理子目录结构
            if args.has_subfolder:
                recons_save_dir = os.path.join(args.recons_dir, paths[i].split("/")[-2])
                dire_save_dir = os.path.join(args.dire_dir, paths[i].split("/")[-2])
            else:
                recons_save_dir = args.recons_dir
                dire_save_dir = args.dire_dir
            
            # 确保输出目录存在
            fn_save = os.path.basename(paths[i])
            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)
            
            # 使用OpenCV保存图像（BGR格式）
            cv2.imwrite(
                f"{dire_save_dir}/{fn_save}",
                cv2.cvtColor(dire[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(
                f"{recons_save_dir}/{fn_save}",
                cv2.cvtColor(recons[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
        
        logger.log(f"have finished {have_finished_images} samples")

    # 等待所有进程完成
    dist.barrier()
    logger.log("finish computing recons & DIRE!")
```



# 五、UNet增加频率模块



## 









