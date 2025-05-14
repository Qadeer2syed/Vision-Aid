# Vision-Aid

**Vision-Aid** is an AI-powered tool designed to generate audio descriptions of video content, enhancing accessibility for visually impaired individuals. By analyzing video frames, it provides contextual audio narratives, enabling users to comprehend visual media through sound.

---

## 🎯 Features

- **Video Frame Analysis**: Utilizes deep learning models to extract meaningful features from video frames.
- **Contextual Audio Narration**: Translates visual elements into coherent audio descriptions.
- **Pretrained Models**: Includes encoder and decoder models for efficient processing.
- **Customizable Training**: Offers scripts to train models on custom datasets.

---

## 🗂️ Project Structure

```
Vision-Aid/
├── Train_data/                 # Directory containing training data
├── IDs.txt                     # List of video IDs used for training/testing
├── Test.py                     # Script to test the model on new data
├── decoder_model_weights.h5    # Pretrained decoder model weights
├── encoder_model.h5            # Pretrained encoder model
├── featureExtractor.py         # Script to extract features from video frames
├── feature_dict2.pickle        # Serialized feature dictionary
├── model_train.py              # Script to train the model
├── tokenizer/                  # Directory containing tokenizer data
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- Required Python packages (see below)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Qadeer2syed/Vision-Aid.git
cd Vision-Aid
```

2. **Install dependencies:**

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Ensure that `requirements.txt` contains all necessary packages.*

---

## 🧪 Usage

### Feature Extraction

Extract features from video frames using the encoder model:

```bash
python featureExtractor.py --input_dir path_to_videos --output_file feature_dict2.pickle
```

### Model Training

Train the model with your dataset:

```bash
python model_train.py --features feature_dict2.pickle --ids IDs.txt
```

### Testing

Generate audio descriptions for new videos:

```bash
python Test.py --video path_to_video
```

*Ensure that the `decoder_model_weights.h5` and `encoder_model.h5` files are present in the project directory.*

---

## 📁 Dataset

The `Train_data/` directory should contain subdirectories or files representing the training videos. The `IDs.txt` file should list the identifiers corresponding to the training data.

---


