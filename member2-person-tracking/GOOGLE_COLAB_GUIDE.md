# Member 2: Multi-Person Tracking Model Training on Google Colab

This guide provides end-to-end instructions for fine-tuning the Member 2 person-detector backbone on Google Colab using a free GPU (NVIDIA T4), evaluating tracking performance, and bringing the trained weights back to your local project.

---

## 1. Architecture Overview (Member 2 Multi-Tracking)

In the shoplifting detection system:
```text
Video Feed → [Member 1: Object Detection] → [Member 2: Multi-Person Tracking] → [Member 3: Behaviour Analysis] → [Member 4: Risk Prediction]
```

### Why does Member 2 have a trainable model?
- **ByteTrack** is an association algorithm (Kalman Filter + 2-stage IoU/CIoU matching). It tracks people across frames and maintains persistent `person_id` values.
- However, ByteTrack **depends directly on the person detector** (`yolov8n-person-track.pt`). Standard generic COCO models frequently lose tracks or misclassify people under retail CCTV angles (overhead perspectives, shelf occlusions, crowded shopping aisles).
- Fine-tuning YOLOv8 on retail frames provides high-recall person detection even with low confidence (0.10–0.25), which allows ByteTrack to maintain continuous tracks through occlusions and reduce ID switches.
- **Why Google Colab?** Training on local CPU takes ~11+ minutes per epoch (~18+ hours for 100 epochs). On Google Colab NVIDIA T4 GPU, 50 epochs takes **~15 to 20 minutes**.

---

## 2. Prepared Files Ready in Your Project

| File / Folder | Purpose |
| --- | --- |
| `notebooks/train_member2_colab.ipynb` | Ready-to-run Jupyter Notebook designed specifically for Google Colab. |
| `member2_dataset_colab.zip` | Pre-packaged dataset archive (17,228 files: 6,823 train images/labels, 1,790 val images/labels, and sample surveillance clips). |
| `package_dataset_for_colab.py` | Helper script to re-pack `Dataset/yolo` anytime you add more images or videos. |

---

## 3. Step-by-Step Training Instructions on Google Colab

### Step 3.1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `member2-person-tracking/notebooks/train_member2_colab.ipynb`.

### Step 3.2: Enable GPU Runtime
1. In Colab menu, go to: **Runtime → Change runtime type**.
2. Under *Hardware accelerator*, select **T4 GPU**.
3. Click **Save**.

### Step 3.3: Upload Dataset Archive
You have two options:
- **Option A (Recommended - Fast & Permanent)**:
  1. Upload `member2_dataset_colab.zip` to your personal **Google Drive** (e.g. root or `My Drive`).
  2. In Colab, uncomment the Google Drive mount cell:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ZIP_SRC = '/content/drive/MyDrive/member2_dataset_colab.zip'
     ```
- **Option B (Direct upload)**:
  1. In the left sidebar of Colab, click the **Folder icon** (`Files`).
  2. Drag and drop `member2_dataset_colab.zip` directly into the files pane.

### Step 3.4: Run the Notebook Cells
Run the cells sequentially (or select **Runtime → Run all**):
1. **GPU Check**: Verifies NVIDIA T4 GPU and PyTorch CUDA.
2. **Install Dependencies**: Installs `ultralytics`, `lapx`, `filterpy`.
3. **Extract Dataset**: Unzips images and labels into `/content/data/dataset`.
4. **Configure `data.yaml`**: Automatically points to the Colab paths.
5. **Train YOLOv8 Detector**: Runs 50 epochs with retail surveillance augmentations (`mosaic=1.0`, `mixup=0.1`, `degrees=10.0`).
6. **Evaluate Metrics**: Displays loss curves, mAP@50, mAP@50-95, precision, recall, and confusion matrix.
7. **Run ByteTrack on Video**: Runs tracking on sample retail video and produces annotated video output.
8. **Export & Download**: Automatically packages `best.pt` (renamed as `yolov8n-person-track.pt`), `results.csv`, and charts, and triggers a download of `member2_trained_model.zip`.

---

## 4. Deploying the Trained Model Back Locally

Once `member2_trained_model.zip` is downloaded to your PC:

1. Extract `yolov8n-person-track.pt` into:
   ```text
   member2-person-tracking/models/yolov8n-person-track.pt
   ```
2. (Optional) Copy `results.csv` and evaluation images into:
   ```text
   member2-person-tracking/runs/detect/person_bytetrack/
   ```

---

## 5. Verifying Tracking Locally

Run the tracking pipeline with your newly trained model:

```powershell
# From member2-person-tracking folder:
python main.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-person-track.pt --output outputs/not_shoplifting1_tracking.json --output-video outputs/not_shoplifting1_tracked.mp4
```

Or run the demo overlay with trajectory trails and analytics:
```powershell
python demo_track.py --video Dataset/videos/not_shoplifting1.mp4 --weights models/yolov8n-person-track.pt --ultralytics-track --output-video outputs/annotated_not_shoplifting1.mp4 --output-json outputs/tracking_log.json
```

Output JSON strictly adheres to the project API contract:
```json
{
  "frame_id": 100,
  "tracks": [
    {
      "person_id": 1,
      "bbox": [210, 145, 340, 520],
      "center": [275, 332]
    }
  ]
}
```
Ready to be ingested by **Member 3 (Suspicious Behaviour Analysis)**!
