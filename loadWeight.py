import timm

try:
    model = timm.create_model('vit_giant_patch14_224_clip_laion2b', pretrained=True)
    # model=timm.create_model('vit_base_patch32_384', pretrained=True)
    print("success!")
except:
    print("fail!")
# model = timm.create_model('vit_giant_patch14_224_clip_laion2b', pretrained=True)