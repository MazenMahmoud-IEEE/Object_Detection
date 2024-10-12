
# Object Detection using R-CNN

This project implements an object detection model using a pre-trained **Faster R-CNN** with a **ResNet-50** backbone from the PyTorch `torchvision` library. The model is trained on the COCO dataset and is capable of detecting 91 common objects.

## Project Flow

1. **Importing Libraries**: The project imports necessary libraries such as `torchvision` for the model, `torch` for handling tensor operations, `cv2` and `PIL` for image processing, and `matplotlib` for visualizations.
   
2. **Prediction Function**:
   - `get_predictions`: This function is used to get predictions from the model. It processes the model's output and filters objects based on a confidence threshold. Bounding boxes and class labels are returned.

3. **Visualization**:
   - The project provides functions (`draw`, `draw_box`) to draw bounding boxes around detected objects on an image, along with class labels and probabilities.

4. **Pre-trained Model**:
   - The model used is `fasterrcnn_resnet50_fpn`, pre-trained on the COCO dataset. The model is used in inference mode with no additional training, and all parameters are frozen.

## Dependencies

- Python 3.x
- PyTorch
- Torchvision
- OpenCV (cv2)
- Pillow (PIL)
- Matplotlib
- Numpy

Install the dependencies using:
```bash
pip install torch torchvision opencv-python pillow matplotlib numpy
```

## How to Use

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Run the notebook or Python script:
   - Use the notebook provided to test the model on your own images.
   - The `get_predictions` function allows you to specify a confidence threshold and filter specific objects.

3. Example usage of the prediction function:
   ```python
   predictions = get_predictions(predicted_output, threshold=0.8)
   draw(predictions, input_image)
   ```

## Results and Examples

The model is capable of detecting objects such as people, cars, animals, etc. Here is an example of a detected object with bounding boxes:

![Example 1](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)
![Example 2](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/340b05c9-50fa-4021-bbbc-5a5096adc622.png)
![Example 3](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/50edd535-d5af-48d5-8c01-43511af626f1.png)
![Example 4](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/5bbba81e-246f-49c2-80f0-8e9e08b6b650.png)
![Example 5](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/8abc7884-6167-4bce-9080-c3c89980efb5.png)
![Example 6](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/9c2620a2-20e4-401a-af1a-66a2b2183ace.png)
![Example 7](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/a148e412-c599-4bb5-85ff-6db8ec9c6504.png)
![Example 8](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/160e38b3-2406-45b7-a1e9-f25e4d7c21a6.png)](https://github.com/MazenMahmoud-IEEE/Object_Detection/blob/main/Results/e5d0009d-da0e-4842-b142-a3cbe04745e6.png)

## Notes

- The model detects 91 different objects from the COCO dataset.
- You can adjust the confidence threshold for filtering predictions.

