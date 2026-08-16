import torch
import numpy as np


def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 10
        elif args.scale == 3:
            args.patch_size = 7
        elif args.scale == 4:
            args.patch_size = 6
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_ssim(img1,img2):
    from skimage.metrics import structural_similarity as ssim
    score = ssim(img1.cpu().numpy().squeeze(0).squeeze(0),img2.cpu().numpy().squeeze(0).squeeze(0),data_range=1)
    return score


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_4d_tensor_as_c_array(tensor, filename, array_name="tensor_array"):
    """
    将四维PyTorch Tensor保存为C语言格式的数组定义
    支持float和uint8_t两种数据类型

    参数:
        tensor: 四维PyTorch Tensor (d0 x d1 x d2 x d3)
        filename: 输出文件名
        array_name: 在C代码中使用的数组名称
    """
    # 确保Tensor在CPU上且为连续内存
    tensor = tensor.detach().cpu().contiguous()

    # 确定数据类型
    if tensor.dtype == torch.float32:
        c_type = "float"
        format_str = "{:.15f}f"  # 修改为15位小数
    elif tensor.dtype == torch.uint8:
        c_type = "uint8_t"
        format_str = "{}"
    else:
        raise ValueError(f"不支持的数据类型: {tensor.dtype}. 仅支持float32和uint8")

    # 获取Tensor的维度
    d0, d1, d2, d3 = tensor.shape

    # 打开文件准备写入
    with open(filename, 'w') as f:
        # 写入数组声明
        f.write(f"const {c_type} {array_name}[{d0}][{d1}][{d2}][{d3}] = {{\n")

        # 遍历所有维度
        for i in range(d0):
            f.write(f"    {{ // d0={i}\n")
            for j in range(d1):
                f.write(f"        {{ // d1={j}\n")
                for k in range(d2):
                    f.write(f"            {{ // d2={k}\n")
                    f.write("                ")

                    # 写入最内层维度值
                    for l in range(d3):
                        # 获取Tensor元素值
                        value = tensor[i, j, k, l].item()

                        # 根据类型格式化值
                        if tensor.dtype == torch.uint8:
                            # uint8_t类型直接写入整数值
                            f.write(format_str.format(int(value)))
                        else:
                            # float类型保留15位小数
                            f.write(format_str.format(value))

                        # 如果不是最后一个元素，添加逗号
                        if l < d3 - 1:
                            f.write(", ")

                        # 每8个元素换行
                        if (l + 1) % 8 == 0 and l < d3 - 1:
                            f.write("\n                ")

                    # 结束最内层维度
                    f.write("\n            }")
                    if k < d2 - 1:
                        f.write(",")
                    f.write("\n")

                # 结束d1维度
                f.write(f"        }}")
                if j < d1 - 1:
                    f.write(",")
                f.write("\n")

            # 结束d0维度
            f.write(f"    }}")
            if i < d0 - 1:
                f.write(",")
            f.write("\n")

        # 结束数组定义
        f.write("};\n")

def save_3d_tensor_as_c_array(tensor, filename, array_name="tensor_array"):
    """
    将四维PyTorch Tensor保存为C语言格式的三维数组定义
    当第一个维度为1时，去掉第一个维度，输出三维数组[d1][d2][d3]

    参数:
        tensor: 四维PyTorch Tensor，第一个维度必须为1 (1 x d1 x d2 x d3)
        filename: 输出文件名
        array_name: 在C代码中使用的数组名称
    """
    # 确保Tensor在CPU上且为连续内存
    tensor = tensor.detach().cpu().contiguous()

    # 检查第一个维度是否为1
    if tensor.shape[0] != 1:
        raise ValueError(f"第一个维度必须为1，当前形状: {tensor.shape}")

    # 确定数据类型
    if tensor.dtype == torch.float32:
        c_type = "float"
        format_str = "{:.15f}f"
    elif tensor.dtype == torch.uint8:
        c_type = "uint8_t"
        format_str = "{}"
    else:
        raise ValueError(f"不支持的数据类型: {tensor.dtype}. 仅支持float32和uint8")

    # 获取Tensor的维度（去掉第一个维度）
    d1, d2, d3 = tensor.shape[1], tensor.shape[2], tensor.shape[3]

    # 打开文件准备写入
    with open(filename, 'w') as f:
        # 写入数组声明（三维数组）
        f.write(f"const {c_type} {array_name}[{d1}][{d2}][{d3}] = {{\n")

        # 遍历所有维度（从第二个维度开始）
        for j in range(d1):
            f.write(f"    {{ // d1={j}\n")
            for k in range(d2):
                f.write(f"        {{ // d2={k}\n")
                f.write("            ")

                # 写入最内层维度值
                for l in range(d3):
                    # 获取Tensor元素值（第一个维度固定为0）
                    value = tensor[0, j, k, l].item()

                    # 根据类型格式化值
                    if tensor.dtype == torch.uint8:
                        # uint8_t类型直接写入整数值
                        f.write(format_str.format(int(value)))
                    else:
                        # float类型保留15位小数
                        f.write(format_str.format(value))

                    # 如果不是最后一个元素，添加逗号
                    if l < d3 - 1:
                        f.write(", ")

                    # 每8个元素换行
                    if (l + 1) % 8 == 0 and l < d3 - 1:
                        f.write("\n            ")

                # 结束最内层维度
                f.write("\n        }")
                if k < d2 - 1:
                    f.write(",")
                f.write("\n")

            # 结束d1维度
            f.write(f"    }}")
            if j < d1 - 1:
                f.write(",")
            f.write("\n")

        # 结束数组定义
        f.write("};\n")

def save_1d_tensor_as_c_array(tensor, filename, array_name="tensor_array"):
    """
    将一维PyTorch Tensor保存为C语言格式的数组定义
    支持float和uint8_t两种数据类型

    参数:
        tensor: 一维PyTorch Tensor (d0)
        filename: 输出文件名
        array_name: 在C代码中使用的数组名称
    """
    # 确保Tensor在CPU上且为连续内存
    tensor = tensor.detach().cpu().contiguous()

    # 检查输入维度
    if len(tensor.shape) != 1:
        raise ValueError(f"输入Tensor必须是一维的，当前形状: {tensor.shape}")

    # 确定数据类型
    if tensor.dtype == torch.float32:
        c_type = "float"
        format_str = "{:.15f}f"
    elif tensor.dtype == torch.uint8:
        c_type = "uint8_t"
        format_str = "{}"
    else:
        raise ValueError(f"不支持的数据类型: {tensor.dtype}. 仅支持float32和uint8")

    # 获取Tensor的长度
    length = tensor.shape[0]

    # 打开文件准备写入
    with open(filename, 'w') as f:
        # 写入数组声明
        f.write(f"const {c_type} {array_name}[{length}] = {{\n")
        f.write("    ")

        # 遍历所有元素
        for i in range(length):
            # 获取Tensor元素值
            value = tensor[i].item()

            # 根据类型格式化值
            if tensor.dtype == torch.uint8:
                # uint8_t类型直接写入整数值
                f.write(format_str.format(int(value)))
            else:
                # float类型保留15位小数
                f.write(format_str.format(value))

            # 如果不是最后一个元素，添加逗号
            if i < length - 1:
                f.write(", ")

            # 每10个元素换行
            if (i + 1) % 10 == 0 and i < length - 1:
                f.write("\n    ")

        # 结束数组定义
        f.write("\n};\n")

def save_1in4d_tensor_as_c_array(tensor, filename, array_name="tensor_array"):
    """
    将四维PyTorch Tensor保存为C语言格式的一维数组定义
    当形状为[1][x][1][1]时，只输出一维数组[x]

    参数:
        tensor: 四维PyTorch Tensor，形状必须为[1][x][1][1]
        filename: 输出文件名
        array_name: 在C代码中使用的数组名称
    """
    # 确保Tensor在CPU上且为连续内存
    tensor = tensor.detach().cpu().contiguous()

    # 检查形状是否符合要求
    if tensor.shape[0] != 1 or tensor.shape[2] != 1 or tensor.shape[3] != 1:
        raise ValueError(f"形状必须为[1][x][1][1]，当前形状: {tensor.shape}")

    # 确定数据类型
    if tensor.dtype == torch.float32:
        c_type = "float"
        format_str = "{:.15f}f"
    elif tensor.dtype == torch.uint8:
        c_type = "uint8_t"
        format_str = "{}"
    else:
        raise ValueError(f"不支持的数据类型: {tensor.dtype}. 仅支持float32和uint8")

    # 获取Tensor的长度
    length = tensor.shape[1]

    # 打开文件准备写入
    with open(filename, 'w') as f:
        # 写入数组声明（一维数组）
        f.write(f"const {c_type} {array_name}[{length}] = {{\n")
        f.write("    ")

        # 遍历所有元素
        for i in range(length):
            # 获取Tensor元素值
            value = tensor[0, i, 0, 0].item()

            # 根据类型格式化值
            if tensor.dtype == torch.uint8:
                # uint8_t类型直接写入整数值
                f.write(format_str.format(int(value)))
            else:
                # float类型保留15位小数
                f.write(format_str.format(value))

            # 如果不是最后一个元素，添加逗号
            if i < length - 1:
                f.write(", ")

            # 每10个元素换行
            if (i + 1) % 10 == 0 and i < length - 1:
                f.write("\n    ")

        # 结束数组定义
        f.write("\n};\n")

def save_weight_tensor_as_c_arrays(tensor, filename_binary, array_name_binary,
                              filename_scale=None, array_name_scale="scale_array"):
    """
    将四维PyTorch Tensor保存为两个C语言格式的数组：
    1. 二值化数组（>0 → 1, ≤0 → 0），保存为 uint8_t
    2. 缩放因子数组（每个 d0 通道的均值缩放因子），保存为 float

    参数:
        tensor: 四维PyTorch Tensor (d0 x d1 x d2 x d3)，必须为 float32
        filename_binary: 二值化数组输出文件名
        array_name_binary: 二值化数组在C代码中的名称
        filename_scale: 缩放因子数组输出文件名（可选，若为 None 则不生成）
        array_name_scale: 缩放因子数组在C代码中的名称（默认 "scale_array"）
    """
    # 确保输入是 float32
    if tensor.dtype != torch.float32:
        raise ValueError(f"输入tensor必须为float32，当前为{tensor.dtype}")

    # 确保Tensor在CPU上且为连续内存
    tensor = tensor.detach().cpu().contiguous()

    # 获取维度
    d0, d1, d2, d3 = tensor.shape

    # === 第一部分：二值化为 0/1 并保存为 uint8_t 数组 ===
    binary_tensor = (tensor > 0).to(torch.uint8)  # >0 → 1, ≤0 → 0

    with open(filename_binary, 'w') as f:
        f.write(f"const uint8_t {array_name_binary}[{d0}][{d1}][{d2}][{d3}] = {{\n")

        for i in range(d0):
            f.write(f"    {{ // d0={i}\n")
            for j in range(d1):
                f.write(f"        {{ // d1={j}\n")
                for k in range(d2):
                    f.write(f"            {{ // d2={k}\n")
                    f.write("                ")

                    for l in range(d3):
                        value = binary_tensor[i, j, k, l].item()
                        f.write(str(value))  # uint8_t 直接写整数

                        if l < d3 - 1:
                            f.write(", ")
                        if (l + 1) % 8 == 0 and l < d3 - 1:
                            f.write("\n                ")

                    f.write("\n            }")
                    if k < d2 - 1:
                        f.write(",")
                    f.write("\n")

                f.write(f"        }}")
                if j < d1 - 1:
                    f.write(",")
                f.write("\n")

            f.write(f"    }}")
            if i < d0 - 1:
                f.write(",")
            f.write("\n")

        f.write("};\n")

    # === 第二部分：计算缩放因子并保存为 float 数组（可选）===
    if filename_scale is not None:
        # 计算缩放因子：对 d1, d2, d3 维度取绝对值后求均值，保留维度
        scaling_factor = torch.mean(
            torch.mean(
                torch.mean(torch.abs(tensor), dim=3, keepdim=True),
                dim=2, keepdim=True
            ),
            dim=1, keepdim=True
        )  # shape: (d0, 1, 1, 1)

        # 转为一维向量 (d0,)
        scale_1d = scaling_factor.squeeze().cpu().contiguous()  # shape: (d0,)

        with open(filename_scale, 'w') as f:
            f.write(f"const float {array_name_scale}[{d0}] = {{\n    ")

            for i in range(d0):
                value = scale_1d[i].item()
                f.write(f"{value:.15f}f")
                if i < d0 - 1:
                    f.write(", ")
                if (i + 1) % 4 == 0 and i < d0 - 1:  # 每4个换行
                    f.write("\n    ")

            f.write("\n};\n")
