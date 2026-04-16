"""
测试 ATCTrack-SemanticAlign 基础模块
"""
import torch
import sys
import os

# 添加项目根目录到路径
project_root = '/workspace/user-data/wyp/ATCTrack_align'
sys.path.insert(0, project_root)

print("=" * 50)
print("测试 Phase 1: 基础模块")
print("=" * 50)

# Test 1: CLIP 文本编码器
print("\n[Test 1] CLIP 文本编码器")
try:
    from lib.models.atctrack.clip_encoder import CLIPTextEncoder

    # 初始化（使用 CPU 避免 GPU 内存问题）
    print("  正在加载 CLIP 模型...")
    clip_encoder = CLIPTextEncoder(model_name='openai/clip-vit-base-patch32', device='cpu')

    # 测试编码
    text_list = ["a white car", "a red ball"]
    text_features = clip_encoder.encode_text_list(text_list)

    print(f"✓ CLIP 编码器加载成功")
    print(f"  输入文本: {text_list}")
    print(f"  输出特征形状: {text_features.shape}")
    print(f"  特征范数: {text_features.norm(dim=-1)}")

except ImportError as e:
    print(f"✗ 缺少依赖: {e}")
    print(f"  提示: transformers 库已安装但可能需要更新")
except Exception as e:
    print(f"✗ CLIP 编码器测试失败: {e}")
    import traceback
    traceback.print_exc()

# Test 2: 语义投影器
print("\n[Test 2] 语义投影器")
try:
    from lib.models.layers.projector import SemanticProjector

    projector = SemanticProjector(input_dim=512, output_dim=512)

    # 测试投影
    dummy_features = torch.randn(4, 512)
    projected = projector(dummy_features)

    print(f"✓ Projector 初始化成功")
    print(f"  输入形状: {dummy_features.shape}")
    print(f"  输出形状: {projected.shape}")
    print(f"  输出范数: {projected.norm(dim=-1)}")

except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ Projector 测试失败: {e}")

# Test 3: RoIAlign 特征提取
print("\n[Test 3] RoIAlign 特征提取")
try:
    from torchvision.ops import roi_align
    from lib.utils.box_ops import box_cxcywh_to_xyxy

    # 模拟特征图和预测框
    B, C, H, W = 2, 512, 16, 16
    feature_map = torch.randn(B, C, H, W)
    bbox_pred = torch.tensor([[0.5, 0.5, 0.3, 0.3],
                              [0.6, 0.4, 0.2, 0.2]])  # (cx, cy, w, h)

    # 转换为 xyxy 格式
    bbox_xyxy = box_cxcywh_to_xyxy(bbox_pred)
    bbox_xyxy_abs = bbox_xyxy * torch.tensor([W, H, W, H])

    # RoIAlign
    batch_indices = torch.arange(B).float().unsqueeze(1)
    rois = torch.cat([batch_indices, bbox_xyxy_abs], dim=1)
    roi_features = roi_align(feature_map, rois, output_size=(7, 7),
                            spatial_scale=1.0, aligned=True)
    instance_features = roi_features.mean(dim=[2, 3])

    print(f"✓ RoIAlign 测试成功")
    print(f"  特征图形状: {feature_map.shape}")
    print(f"  预测框: {bbox_pred}")
    print(f"  实例特征形状: {instance_features.shape}")

except Exception as e:
    print(f"✗ RoIAlign 测试失败: {e}")

print("\n" + "=" * 50)
print("基础模块测试完成！")
print("=" * 50)
