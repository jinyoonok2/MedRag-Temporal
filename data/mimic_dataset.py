import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class MimicUnifiedDataset(Dataset):
    def __init__(self, tokenizer, context_length, transforms, merged_df, download_path):
        """
        Dataset class for MIMIC-CXR unified data format.

        Args:
            tokenizer: Tokenizer with __call__(text, context_length) -> token tensor
            context_length: Max token length
            transforms: torchvision-style transform (PIL.Image -> Tensor)
            merged_df: DataFrame with metadata
            download_path: root path to .pt, .txt, .json files
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.transforms = transforms
        self.base_dir = str(download_path)
        self.df = merged_df.copy()

        self.label_cols = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
            "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
            "Pneumothorax", "Support Devices"
        ]

        self.df[self.label_cols] = self.df[self.label_cols].fillna(-1)
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for _, row in self.df.iterrows():
            study_id = str(row["study_id"])
            dicom_id = str(row["dicom_id"])
            study_path = os.path.join(self.base_dir, study_id)

            image_path = os.path.join(study_path, f"{dicom_id}.jpg")
            report_txt = os.path.join(study_path, f"s{study_id}.txt")
            refined_report_txt = os.path.join(study_path, f"s{study_id}_refined.txt")
            graph_json = os.path.join(study_path, f"s{study_id}_graph.json")

            missing = []
            if not os.path.exists(image_path):
                missing.append("image_path")
            if not os.path.exists(report_txt):
                missing.append("report_txt")
            if missing:
                raise FileNotFoundError(
                    f"Missing files for study {study_id}, dicom {dicom_id}: {missing}"
                )

            label_tensor = torch.tensor(
                [float(row[col]) for col in self.label_cols],
                dtype=torch.float32
            )

            samples.append({
                "study_id": study_id,
                "image_path": image_path,
                "report_txt_path": report_txt,
                "refined_report_txt_path": refined_report_txt,
                "graph_json_path": graph_json,
                "label": label_tensor,
                "study_path": study_path,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load JPG -> PIL -> transforms -> Tensor
        pil_image = Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transforms(pil_image)

        with open(sample["report_txt_path"], "r", encoding="utf-8", errors="ignore") as f:
            report_txt = f.read().strip()

        if os.path.exists(sample["refined_report_txt_path"]):
            with open(sample["refined_report_txt_path"], "r", encoding="utf-8", errors="ignore") as f:
                refined_report_txt = f.read().strip()
        else:
            refined_report_txt = ""

        # Tokenize refined (no fallback)
        refined_report_token = self.tokenizer(
            refined_report_txt,
            context_length=self.context_length
        ).squeeze(0)

        return {
            "study_id": sample["study_id"],
            "study_path": sample["study_path"],
            "image_pt": image_tensor,  # same key name as before
            "image_path": sample["image_path"],
            "report_txt": report_txt,
            "report_txt_path": sample["report_txt_path"],
            "refined_report_txt": refined_report_txt,
            "refined_report_txt_path": sample["refined_report_txt_path"],
            "refined_report_token": refined_report_token,
            "label": sample["label"],
            "graph_json_path": sample["graph_json_path"],
        }
