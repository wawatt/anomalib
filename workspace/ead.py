import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="openvino.runtime")

import torch
torch.set_float32_matmul_precision('high')

from anomalib.data import MVTecAD, Kolektor
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import Patchcore,EfficientAd,Padim
from anomalib.engine import Engine
from anomalib.metrics import Evaluator, AUROC, F1Score, PRO
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, InterpolationMode

if __name__ == '__main__':


    evaluator = Evaluator(
        val_metrics=[
            AUROC(fields=["pred_score", "gt_label"],   prefix="Image-level  "),     # Image-level detection
            # F1Score(fields=["pred_label", "gt_label"], prefix="I"),
            AUROC(fields=["anomaly_map", "gt_mask"],   prefix="Pixel-level  "),     # Pixel-level detection accuracy
            # F1Score(fields=["pred_mask", "gt_mask"],   prefix="P"),     # Segmentation quality
            PRO(fields=["anomaly_map", "gt_mask"],     prefix="Pixel-level     ")        # Region-based evaluation
        ],
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"],   prefix="Image-level   "),     # Image-level detection
            F1Score(fields=["pred_label", "gt_label"], prefix="Image-level "),
            AUROC(fields=["anomaly_map", "gt_mask"],   prefix="Pixel-level   "),     # Pixel-level detection accuracy
            F1Score(fields=["pred_mask", "gt_mask"],   prefix="Pixel-level "),       # Segmentation quality
            PRO(fields=["anomaly_map", "gt_mask"],     prefix="Pixel-level     ")    # Region-based evaluation
        ]
    )

    # Initialize components
    # datamodule = Kolektor(
    #     num_workers=1,
    #     train_batch_size=1,
    #     # eval_batch_size=4,
    #     seed=42)
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="bottle",
        num_workers=1,
        train_batch_size=1,
        # eval_batch_size=4,
        seed=42)

    transform=Compose([
        Resize(size=[320, 320], interpolation=InterpolationMode.BILINEAR, antialias=True)
    ])

    pre_processor = PreProcessor(transform=transform)
    model = EfficientAd(
        imagenet_dir="datasets/subsetOfKylbergTextureDataset-6classes-40samples",
        evaluator=evaluator, 
        pre_processor=pre_processor
    )
    # model = Patchcore(
    #     pre_processor=pre_processor,
    #     backbone="efficientvit_b3", 
    #     layers=["stages.2", "stages.3"], 
    #     pre_trained=True, 
    #     coreset_sampling_ratio=0.1, 
    #     num_neighbors=9,
    #     evaluator=evaluator)
    # model = Padim(
    #     pre_processor=pre_processor,
    #     backbone="efficientvit_b3", 
    #     layers=["stages.1", "stages.2"], 
    #     n_features=384,
    #     evaluator=evaluator
    # )    
    kwargs = {"log_every_n_steps": 1}

    callbacks = [
        ModelCheckpoint(
            mode="max",
            monitor="Pixel-level     PRO",
        ),
        EarlyStopping(
            monitor="Pixel-level     PRO",
            mode="max",
            patience=3,
        ),
    ]
    
    engine = Engine(max_epochs=1,callbacks=callbacks) # logger

    # Train the model
    engine.train(datamodule=datamodule, model=model)
    engine.export(
        model=model,
        export_type="onnx",
        # input_size=(320, 320),  # Adjust based on your needs
    )