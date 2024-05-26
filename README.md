## Text Analysis

This repository contains a collection of notebooks and resources for various NLP tasks using different architectures and frameworks. Each notebook focuses on a specific task and demonstrates the implementation using appropriate datasets and models.

## Notebooks

### 1. Bidirectional-Stacked-LSTM.ipynb
- **Task**: Text Classification using RNN-LSTM
- **Direction**: Bi-directional
- **Layer Type**: Stacked
- **Framework**: TensorFlow with Keras
- **Dataset**: IMdB movie review

This notebook demonstrates text classification using a bidirectional stacked LSTM architecture. It uses the IMdB movie review dataset for training and evaluation.

### 2. LanguageModelling.ipynb
- **Task**: Language Modeling (LM)
- **Architecture**: RNN / LSTM
- **Framework**: TensorFlow with Keras

This notebook focuses on language modeling using RNN / LSTM architectures. It demonstrates the training process and evaluates the model's performance.

### 3. MaskedLanguageModelling.ipynb
- **Task**: Masked Language Modeling (MLM)
- **Architecture**: DistilRoBERTa
- **Dataset**: ELI5 dataset
- **Framework**: PyTorch using HuggingFace

This notebook covers masked language modeling using the DistilRoBERTa architecture and the ELI5 dataset. It utilizes the HuggingFace library for model implementation and training.

### 4. NamedEntityRecognition.ipynb
- **Task**: Sequence Labeling - Named Entity Recognition (NER)
- **Architecture**: Transformer
- **Dataset**: CoNLL 2003
- **Framework**: Keras

This notebook demonstrates named entity recognition using transformer architectures. It uses the CoNLL 2003 dataset and Keras for model implementation.

### 5. QuestionAnsweringSystem.ipynb
- **Task**: Question Answering (QA)
- **Architecture**: BERT - Transformers
- **Dataset**: SQuAD
- **Framework**: TensorFlow with Keras / PyTorch (HuggingFace)

This notebook focuses on building a question-answering system using BERT and transformer models. The SQuAD dataset is used for training and evaluation, with implementations available in both TensorFlow with Keras and PyTorch using HuggingFace.

### 6. Single-layer-LSTM.ipynb
- **Task**: Text Classification using RNN-LSTM
- **Direction**: Sequential
- **Layer Type**: Single
- **Framework**: TensorFlow with Keras
- **Dataset**: IMdB movie review

This notebook provides a simpler text classification example using a single-layer RNN-LSTM architecture. It also uses the IMdB movie review dataset.

### 7. TextClassificationUsingTransformers.ipynb
- **Task**: Sequence Classification
- **Architecture**: BERT - Transformers
- **Dataset**: IMdB
- **Framework**: TensorFlow with Keras / PyTorch (HuggingFace)

This notebook covers text classification using BERT transformer models on the IMdB dataset. It includes implementations in TensorFlow with Keras and PyTorch using HuggingFace.

### 8. XOR-Scikit-Learn.ipynb
This notebook demonstrates the XOR problem using Scikit-Learn. It provides a basic example of solving a classic machine learning problem.

## Additional Files

### wonderland.txt
This text file is used for language modeling tasks. It contains text data for training language models.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/selcia25/text-analysis.git
   cd text-analysis
   ```
2. **Run the Notebooks**:
   Open the notebooks using Jupyter or any other compatible environment and run the cells to train and evaluate the models.

## Requirements

- Python
- TensorFlow
- Keras
- PyTorch
- HuggingFace Transformers
- Scikit-Learn
- Jupyter Notebook

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to the contributors and the open-source community for providing the datasets and frameworks used in these notebooks.
