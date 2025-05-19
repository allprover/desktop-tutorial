# 一、AIGC类管理

## 1. 模型名称映射数字标签 

```python
# AIGC类别映射
CLASS2LABEL_MAPPING = {
    'real': 0,  # 正常图像, MSCOCO, ImageNet等
    'ldm-text2im-large-256': 1,  # 'CompVis/ldm-text2im-large-256': 'Latent Diffusion',  # Latent Diffusion 基础版本
    'stable-diffusion-v1-4': 2,  # 'CompVis/stable-diffusion-v1-4': 'Stable Diffusion',  # 现实版本
    'stable-diffusion-v1-5': 3,  # 'runwayml/stable-diffusion-v1-5': 'Stable Diffusion',  # 现实版本
    'stable-diffusion-2-1': 4,
    'stable-diffusion-xl-base-1.0': 5,
    'stable-diffusion-xl-refiner-1.0': 6,
    'sd-turbo': 7,
    'sdxl-turbo': 8,
    'lcm-lora-sdv1-5': 9,
    'lcm-lora-sdxl': 10,
    'sd-controlnet-canny': 11,
    'sd21-controlnet-canny': 12,
    'controlnet-canny-sdxl-1.0': 13,
    'stable-diffusion-inpainting': 14,
    'stable-diffusion-2-inpainting': 15,
    'stable-diffusion-xl-1.0-inpainting-0.1': 16,
}
LABEL2CLASS_MAPPING = {CLASS2LABEL_MAPPING.get(key): key for key in CLASS2LABEL_MAPPING.keys()}
GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
                 'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
                 'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']
```

**作用**

将AIGC生成模型的名称映射为数字标签（0-16）



## 2. 查找键值对

```python
LABEL2CLASS_MAPPING = {CLASS2LABEL_MAPPING.get(key): key for key in CLASS2LABEL_MAPPING.keys()}
```

**作用**

通过数字标签反向查找对应的模型名称。



## 3. 图像路径列表

```python
GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
                 'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
                 'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']
```

**作用**

列出AI生成图像的路径或标识符，可能用于数据集中的子目录或文件命名。



# 二、真实和AI图像数据集（DIRE）

## 1. 函数参数

```python
def load_DRCT_2M(
    real_root_path='/disk4/chenby/dataset/MSCOCO',  # 真实图像的根目录
    fake_root_path='/disk4/chenby/dataset/AIGC_MSCOCO',  # AI生成图像的根目录
    fake_indexes='1,2,3,4,5,6',  # 指定要加载的AI生成模型的标签（对应CLASS2LABEL_MAPPING中的值）
    phase='train',  # 数据加载阶段：'train'、'val'或'test'
    val_split=0.1,  # 验证集比例（默认10%）
    seed=2022  # 随机种子，确保数据划分可复现
):
```



## 2. 处理 fake_indexes 字符串

```python
fake_indexes = [int(index) for index in fake_indexes.split(',')]
```

- 将输入的字符串（如`'1,2,3,4,5,6'`）转换为整数列表 `[1, 2, 3, 4, 5, 6]`，表示需要加载的AI生成模型的标签。





## 3. 加载两大标签图像

### 3.1 不是 test 阶段

**真实图像**

```pytHon
real_paths = sorted(glob.glob(f"{real_root_path}/train2017/*.*"))  # 加载所有训练图像
real_labels = [0 for _ in range(len(real_paths))]  # 标签为0（真实图像）
real_paths, real_labels = split_data(real_paths, real_labels, val_split=val_split, phase=phase, seed=seed)
```

**AI 生成图像**

```python
for i, index in enumerate(fake_indexes):
    fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/train2017/*.*"))
    fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]  # 标签从1开始递增
    fake_paths_t, fake_labels_t = split_data(fake_paths_t, fake_labels_t, val_split=val_split, phase=phase, seed=seed)
    fake_paths += fake_paths_t
    fake_labels += fake_labels_t
```

- 遍历 `fake_indexes`，加载每个AI模型生成的图像路径（如 `fake_root_path/stable-diffusion-v1-4/train2017/`）。
- 标签从 `1` 开始递增（`i + 1`），与 `CLASS2LABEL_MAPPING` 中的模型标签对应。
- 同样调用 `split_data` 划分数据。



### 3.2 是 test 阶段

**真实图像**

```
real_paths = sorted(glob.glob(f"{real_root_path}/val2017/*.*"))  # 加载所有验证图像作为测试集
real_labels = [0 for _ in range(len(real_paths))]
```

**AI生成图像**

```
for i, index in enumerate(fake_indexes):
    fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/val2017/*.*"))
    fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
    fake_paths += fake_paths_t
    fake_labels += fake_labels_t
```

- 从 `val2017/` 加载图像路径，不进行划分，直接合并。



## 4. 合并数据

```
image_paths = real_paths + fake_paths
labels = real_labels + fake_labels
```

- 合并真实图像和AI生成图像的路径及标签。





# 三、GenImage数据集（DIRE)

## 1. 函数参数

```python
def load_GenImage(
    root_path='/disk1/chenby/dataset/AIGC_data/GenImage',  # 数据集根目录
    phase='train',  # 数据加载阶段：'train'、'val'或'test'
    seed=2023,  # 随机种子，确保数据划分可复现
    indexes='1,2,3,4,5,6,7,8',  # 指定要加载的数据集子集索引（对应GenImage_LIST中的位置）
    val_split=0.1  # 验证集比例（默认10%）
):
```

## 2. 处理 indexes 字符串

```
indexes = [int(i) - 1 for i in indexes.split(',')]
```



## 3. 选择数据集

```
dir_list = GenImage_LIST
selected_dir_list = [dir_list[i] for i in indexes]
```



## 4. 初始化数据容器

```
real_images, real_labels, fake_images, fake_labels = [], [], [], []
dir_phase = 'train' if phase != 'test' else 'val'
```





# 四、对比数据集（DRCT）

## 1. 函数参数

```python
def load_pair_data(
    root_path,  # 真实图像的根目录（或包含真实和修复路径的字符串）
    fake_root_path=None,  # AI生成图像的根目录（可选）
    phase='train',  # 数据加载阶段：'train'、'val'或'test'
    seed=2023,  # 随机种子，确保数据划分可复现
    fake_indexes='1',  # 指定AI生成模型的标签（对应CLASS2LABEL_MAPPING或GenImage_LIST中的索引）
    inpainting_dir='full_inpainting'  # 修复图像的目录名称
):
```



## 2. fake_root_path 为 None

```python
if fake_root_path is None:
    assert len(root_path.split(',')) == 2
    root_path, rec_root_path = root_path.split(',')[:2]
    image_paths = sorted(glob.glob(f"{root_path}/*.*"))
    rec_image_paths = sorted(glob.glob(f"{rec_root_path}/*.*"))
    assert len(image_paths) == len(rec_image_paths)
    total_paths = []
    for image_path, rec_image_path in zip(image_paths, rec_image_paths):
        total_paths.append((image_path, rec_image_path))
    print(f'Pair data-{phase}:{len(total_paths)}.')
    return total_paths
```



- **用途**：当不需要加载AI生成图像时（如推理或特征提取），直接加载原始图像和修复图像的路径对。



## 3. 路径验证和初始化

```
assert (len(root_path.split(',')) == 2 and len(fake_root_path.split(',')) == 2) or \
       (root_path == fake_root_path and 'GenImage' in root_path)
```

- 检查路径格式：
  - 如果路径是逗号分隔的字符串，则必须包含两部分（原始路径和修复路径）。
  - 如果 `root_path` 和 `fake_root_path` 相同且路径中包含 `GenImage`，则认为是GenImage数据集。



## 4. 处理不同数据集类型

这里我们只看 GenImage，里面的 fake_name 在 GenImage_LIST 可以找到

```python
elif 'DR/GenImage' in root_path:
    phase_mapping = {'train': 'train', 'val': 'train', 'test': 'val'}
    fake_indexes = int(fake_indexes)
    assert 1 <= fake_indexes <= 8 and inpainting_dir in ['inpainting', 'inpainting2', 'inpainting_xl']
    fake_name = GenImage_LIST[fake_indexes-1]
    real_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/crop'
    real_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/{inpainting_dir}'
    fake_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/crop'
    fake_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/{inpainting_dir}'
    print(f'fake_name:{fake_name}')
```



- 目录结构：
  - 真实图像：`{root_path}/{子集名称}/train/nature/crop/*.*`。
  - 修复图像：`{root_path}/{子集名称}/train/nature/inpainting/*.*`。
  - AI生成图像：`{root_path}/{子集名称}/train/ai/crop/*.*`。
  - AI修复图像：`{root_path}/{子集名称}/train/ai/inpainting/*.*`。





# 五、Dataset 类

