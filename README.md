# Machine Learning – Assignment 2

Chosen option: Computer Vision

## Students:
Kristen Karsburg Arguello – 22103087  
Ramiro Nilson Barros – 22111163  
Vinícius Conte Turani – 22106859

## How to reproduce the results:
1) Install dependencies:
```sh
pip install -r requirements.txt
```

2) Train the model:
```sh
python3 model_training.py
```
| This generates the file `checkpoint.pth` containing the final weights.

3) Run the demo application:
```sh
python3 camera.py
```
*Note*: On the first run, the system will request webcam access permission.  
If denied or not yet granted, the program will exit. Just allow access and re-run the command above.

4) Use the demo: 
- A webcam window will appear.
- Position the person's face within the frame and press the SPACEBAR to capture a frame.
- The model will classify the image as either “woman” or “man” and display the result on the screen.
- To close the application, press ESC.

### Extras — how to reproduce the graphs
1. **Training output files**  
   When training the model, three files are saved in the `apresentacao` folder:
   - `confusion_matrix.png` – normalized confusion matrix  
   - `loss_accuracy.png` – loss and accuracy curves per epoch  
   - `feature_maps_demo.png` – visualization of feature maps for the first two samples  
   
2. **Data transformation inspection**  
   To view before/after examples from the data augmentation pipeline, run:  
   ```bash
   python documenting_dataset.py
   ```

#### Transformations applied, in order:
**Geometry**
- Rotation ±30° (50%)
- Scaling ±20% (50%)
- Shift ±10% (50%)
- Horizontal flip (50%)

**Occlusion and blur**
- CoarseDropout (1 hole, 20 × 20 px, value 128, 50%)
- Gaussian blur (3 px, 30%)

**Color and brightness** *(one of the following, 80% probability)*
- Brightness/contrast ±20% (70%)
- Hue ±15°, Saturation ±25%, Value ±15% (70%)
- Channel shuffle (30%)

**Noise**
- Gaussian noise with σ² between 10 and 50 (30%)

**Post-processing**
- Resize to 64 × 64
- Normalize (mean = 0.5, std = 0.5)
- Convert to tensor (ToTensorV2)

---

## Quick project overview
- **Goal:** classify 64 × 64 images as *woman* or *man* using a lightweight CNN (~2M parameters).  
- **Architecture (`SimpleCNN`):** 4 blocks `Conv → BN → ReLU → MaxPool`, from 3×64×64 to 128×4×4; then `Flatten → Dropout → Linear → ReLU → Dropout → Linear` to logits.  
- **Augmentation:** rotations, affine (translate/scale/shear), H/V flips, `ColorJitter`, Gaussian noise, `RandomErasing`; all followed by `Normalize([0.5], [0.5])`.  
- **Training:** Adam (LR = 1e-3) + `StepLR` (γ = 0.1 every 3 epochs), `CrossEntropyLoss`, 10 default epochs, checkpoint saved to `checkpoint.pth`.  
- **Evaluation:** accuracy on *test set* + normalized confusion matrix (saved to `apresentacao/confusion_matrix.png`).  
- **Extras:** *forward hook* saves feature maps from the last conv block; script prints mean/σ of the dataset and optionally of webcam captures for domain analysis.

## Stack:
- **PyTorch / TorchVision** – deep learning backbone: defines the CNN (`nn.Module`), runs training/inference on CPU or GPU, and provides base transform utilities.
- **Albumentations (+ ToTensorV2)** – advanced data augmentation engine; applies rotation, affine, noise, blur, etc., and returns PyTorch-compatible tensors.
- **OpenCV** – real-time video capture, digital zoom, and overlay of results in the
