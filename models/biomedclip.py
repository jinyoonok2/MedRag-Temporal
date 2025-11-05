from utils import get_pretrained_model, get_tokenizer, get_preprocess_transform
import torch

class BioMedCLIP:
    def __init__(self, model_cfg, use_best=True):
        self.model = get_pretrained_model(model_cfg)
        self.tokenizer = get_tokenizer(model_cfg)
        self.transforms = get_preprocess_transform(model_cfg)

        # 👉 Load best checkpoint if requested
        if use_best and getattr(model_cfg, "best_pt", None):
            ckpt = torch.load(model_cfg.best_pt, map_location="cpu")

            # Accept common layouts: raw state_dict, or wrapped in keys
            if isinstance(ckpt, dict):
                for key in ["state_dict", "model", "model_state_dict"]:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        ckpt = ckpt[key]
                        break

            # Strip "module." if saved from DataParallel
            if isinstance(ckpt, dict):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

            missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
            print(f"✅ Loaded weights from: {model_cfg.best_pt}")
            if missing:
                print(f"⚠️ Missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
            if unexpected:
                print(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

        # 🔒 Apply freeze settings (still honored whether or not we loaded a ckpt)
        if model_cfg.vision_freeze:
            print("🧊 Freezing visual trunk (encoder)")
            for p in self.model.visual.trunk.parameters():
                p.requires_grad = False

        if model_cfg.vision_proj_freeze:
            print("🧊 Freezing visual projection head")
            for p in self.model.visual.head.parameters():
                p.requires_grad = False

        if model_cfg.text_freeze:
            print("🧊 Freezing text encoder (BERT)")
            for p in self.model.text.transformer.parameters():
                p.requires_grad = False

        if model_cfg.text_proj_freeze:
            print("🧊 Freezing text projection head")
            for p in self.model.text.proj.parameters():
                p.requires_grad = False

        # 📊 Print parameter summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"\n📊 Model Parameter Summary:")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters:    {frozen_params:,} ({frozen_params / total_params:.2%})")

        # 🛠️ Training config (kept even if use_best=True)
        self.train_cfg = {
            "batch_size": model_cfg.batch_size,
            "num_workers": model_cfg.num_workers,
            "learning_rate": model_cfg.learning_rate,
            "weight_decay": model_cfg.weight_decay,
            "epochs": model_cfg.epochs,
            "context_length": model_cfg.context_length,
            "weights_dir": model_cfg.weights_dir,
            "best_pt": model_cfg.best_pt,
            "last_pt": model_cfg.last_pt,
            "vision_freeze": model_cfg.vision_freeze,
            "vision_proj_freeze": model_cfg.vision_proj_freeze,
            "text_freeze": model_cfg.text_freeze,
            "text_proj_freeze": model_cfg.text_proj_freeze,
        }

        print(f"\n🧾 Training Configuration:")
        for key, val in self.train_cfg.items():
            print(f"   {key:20}: {val}")
