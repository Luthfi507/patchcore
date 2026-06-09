import timm  # noqa
import torchvision.models as models  # noqa

# Gunakan weights="DEFAULT" (torchvision >= 0.13) sebagai pengganti pretrained=True
# yang sudah deprecated. Fallback ke pretrained=True untuk versi lama.
try:
    _WEIGHT_PARAM = "weights='DEFAULT'"
    models.resnet50(weights="DEFAULT")  # test apakah API baru tersedia
except TypeError:
    _WEIGHT_PARAM = "pretrained=True"

_BACKBONES = {
    "alexnet":           f"models.alexnet({_WEIGHT_PARAM})",
    "resnet50":          f"models.resnet50({_WEIGHT_PARAM})",
    "resnet101":         f"models.resnet101({_WEIGHT_PARAM})",
    "resnext101":        f"models.resnext101_32x8d({_WEIGHT_PARAM})",
    "vgg11":             f"models.vgg11({_WEIGHT_PARAM})",
    "vgg19":             f"models.vgg19({_WEIGHT_PARAM})",
    "vgg19_bn":          f"models.vgg19_bn({_WEIGHT_PARAM})",
    "wideresnet50":      f"models.wide_resnet50_2({_WEIGHT_PARAM})",
    "wideresnet101":     f"models.wide_resnet101_2({_WEIGHT_PARAM})",
    "bninception":       'pretrainedmodels.__dict__["bninception"]'
                         '(pretrained="imagenet", num_classes=1000)',
    "resnet200":         'timm.create_model("resnet200", pretrained=True)',
    "resnest50":         'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit":   'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k":   'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit":  'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k":  'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit":  'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k":  'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384":  'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101":      'timm.create_model("resnetv2_101", pretrained=True)',
    "mnasnet_100":       'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1":        'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1":        'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121":       'timm.create_model("densenet121", pretrained=True)',
    "densenet201":       'timm.create_model("densenet201", pretrained=True)',
    "inception_v4":      'timm.create_model("inception_v4", pretrained=True)',
    "vit_small":         'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base":          'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large":         'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50":           'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base":     'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled":'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base":     'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large":    'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7":   'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5":   'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3":   'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1":   'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m":  'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l":  'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a":  'timm.create_model("efficientnet_b3a", pretrained=True)',
}


def load(name):
    return eval(_BACKBONES[name])