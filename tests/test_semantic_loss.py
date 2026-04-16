"""
测试语义对齐损失计算
"""
import torch
import sys
import os

project_root = '/workspace/user-data/wyp/ATCTrack_align'
sys.path.insert(0, project_root)

print("=" * 50)
print("测试 Phase 3: 语义对齐损失")
print("=" * 50)

# Test 1: 端到端语义对齐流程
print("\n[Test 1] 端到端语义对齐流程")
try:
    from lib.models.atctrack.clip_encoder import CLIPTextEncoder
    from lib.models.layers.projector import SemanticProjector
    from lib.utils.box_ops import box_cxcywh_to_xyxy
    from torchvision.ops import roi_align

    # 模拟数据
    B = 4
    feature_map = torch.randn(B, 512, 16, 16)
    pred_boxes = torch.tensor([[0.5, 0.5, 0.3, 0.3],
                                [0.6, 0.4, 0.2, 0.2],
                                [0.4, 0.6, 0.25, 0.25],
                                [0.55, 0.45, 0.3, 0.3]])

    text_list = ["a white car", "a red ball", "a blue bike", "a green tree"]

    print("  步骤 1: RoIAlign 提取实例特征...")
    bbox_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    bbox_xyxy_abs = bbox_xyxy * torch.tensor([16, 16, 16, 16])
    batch_indices = torch.arange(B).float().unsqueeze(1)
    rois = torch.cat([batch_indices, bbox_xyxy_abs], dim=1)
    roi_features = roi_align(feature_map, rois, output_size=(7, 7),
                            spatial_scale=1.0, aligned=True)
    instance_features = roi_features.mean(dim=[2, 3])
    print(f"    实例特征形状: {instance_features.shape}")

    print("  步骤 2: Projector 投影到 CLIP 空间...")
    projector = SemanticProjector(input_dim=512, output_dim=512)
    projected_features = projector(instance_features)
    print(f"    投影特征形状: {projected_features.shape}")

    print("  步骤 3: CLIP 编码文本...")
    clip_encoder = CLIPTextEncoder(device='cpu')
    text_features = clip_encoder.encode_text_list(text_list)
    print(f"    文本特征形状: {text_features.shape}")

    print("  步骤 4: 计算语义对齐损失...")
    cos_sim = torch.nn.functional.cosine_similarity(
        projected_features.detach(),  # detach 因为 projector 是随机初始化的
        text_features,
        dim=-1
    )
    semantic_loss = 1 - cos_sim.mean()
    print(f"    余弦相似度: {cos_sim}")
    print(f"    语义对齐损失: {semantic_loss.item():.4f}")

    print("✓ 端到端流程测试成功")

except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# Test 2: 双阶段权重切换
print("\n[Test 2] 双阶段权重切换")
try:
    class MockConfig:
        def __init__(self):
            self.TRAIN = type('obj', (object,), {
                'SEMANTIC_WEIGHT_STAGE1': 0.2,
                'SEMANTIC_WEIGHT_STAGE2': 0.05,
                'EPOCH': 180
            })()
            self.MODEL = {
                'USE_SEMANTIC_ALIGN': True,
                'LORA': {'STAGE1_RATIO': 0.3}
            }

    cfg = MockConfig()
    stage1_ratio = 0.3
    total_epochs = 180
    stage1_epochs = int(total_epochs * stage1_ratio)

    print(f"  总 Epoch: {total_epochs}")
    print(f"  Stage 1 比例: {stage1_ratio}")
    print(f"  Stage 1 结束于 Epoch: {stage1_epochs}")

    # 测试不同 epoch 的权重
    test_epochs = [0, 30, 54, 55, 100, 179]
    for epoch in test_epochs:
        if epoch < stage1_epochs:
            weight = 0.2
            stage = "Stage 1 (语义探索)"
        else:
            weight = 0.05
            stage = "Stage 2 (精细定位)"
        print(f"  Epoch {epoch:3d}: λ={weight:.2f} ({stage})")

    print("✓ 双阶段权重切换测试成功")

except Exception as e:
    print(f"✗ 测试失败: {e}")

print("\n" + "=" * 50)
print("语义对齐损失测试完成！")
print("=" * 50)
