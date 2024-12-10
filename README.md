# Deep Learning and Generative AI
## [Dr. Elias Jacob de Menezes Neto](https://docente.ufrn.br/elias.jacob)
Welcome to the Introductory course on Deep Learning and Generative AI. This repository contains the code and resources for the course. Our syllabus covers a wide range of topics in these fields. Yeah, it is a lot, I know. But we will have fun!

> This course will be taught at [Instituto Metrópole Digital/UFRN](https://portal.imd.ufrn.br/portal/)

## Deep Learning

In the Deep Learning section, we will explore the following topics:

- **Perceptron and Activation Functions**: Learn about the building blocks of neural networks and how they enable learning complex patterns.
- **Multilayer Perceptrons**: Understand how multiple layers of perceptrons can be combined to create powerful deep learning models.
- **Backpropagation, Stochastic Gradient Descent, and Optimizers**: Dive into the training process of neural networks, including the backpropagation algorithm, stochastic gradient descent, and various optimization techniques.
- **Basics of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs)**: Get an introduction to two popular architectures used in deep learning: CNNs for image and video processing, and RNNs for sequential data analysis.
- **Overview of Computer Vision**: Explore various tasks in computer vision, such as image segmentation, enhancement, and feature extraction.
- **Tools: PyTorch and TensorFlow/Keras**: Gain hands-on experience with popular deep learning frameworks. We'll focus on PyTorch.
- **Practical Exercises**: Apply your knowledge through practical exercises and projects.

## Generative AI

The Generative AI section will cover the following topics:

- **Transformer Architecture**: Learn about the revolutionary transformer architecture, including self-attention mechanisms and the encoder-decoder structure. Understand the impact of models like BERT.
- **Embeddings**: Dive into the concept of embeddings and their role in representing text, images, and other data types in generative models.
- **Large Language Models (LLMs)**: Explore the world of foundation models, fine-tuning techniques, and prompt engineering for generating human-like text.
- **Retrieval Augmented Generation (RAG)**: Understand how retrieval-based methods can enhance the performance and capabilities of generative models.
- **Tools: Hugging Face, PyTorch/TensorFlow, Langchain**: Get hands-on experience with popular libraries and frameworks for building and deploying generative AI models.

Throughout the course, you will have the opportunity to work on practical exercises and projects to solidify your understanding of deep learning and generative AI concepts. I will provide code examples and resources to support your learning and help you build your projects.

## Getting Started

### Prerequisites

To get started with the course, ensure that you have the following prerequisites:

- You know Python and have experience with Machine Learning and Natural Language Processing concepts.
- Access to a machine with a GPU (recommended) or Google Colab for running computationally intensive tasks.
- An account with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization.
- Installation of [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) for managing Python environments.

To install Poetry, you can use the following command:
```shell
curl -sSL https://install.python-poetry.org | python3 -
``` 

> Note: Some parts of the code may require a GPU for efficient execution. If you don't have access to a GPU, consider using Google Colab.

### Installation

Follow these steps to set up the environment and dependencies:

1. **Download the Repository**:
    ```shell
    git clone https://github.com/eliasjacob/deep_learning_gen_ai.git
    cd deep_learning_gen_ai
    ```

2. **Run the Download Script**:
    ```shell
    bash download_datasets_and_binaries.sh
    ```

3. **Install Ollama**:
   Download and install Ollama from [here](https://ollama.com/download).

4. **Download LLama 3.1**:
    ```bash
    ollama pull llama3.1
    ```

5. **Install Dependencies**:
 - For GPU support:
    ```shell
    poetry install --sync -E cuda --with cuda
    poetry shell
    ```
    
- For CPU-only support:
    ```shell
    poetry install --sync -E cpu
    poetry shell
    ```

6. **Authenticate Weights & Biases**:
    ```bash
    wandb login
    ```

## Using VS Code Dev Containers

This repository is configured to work with Visual Studio Code Dev Containers, providing a consistent and isolated development environment. To use this feature:

1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.

2. Clone this repository to your local machine (if you haven't already):

   ```shell
   git clone https://github.com/eliasjacob/deep_learning_genai.git
   ```

3. Open the cloned repository in VS Code.

4. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container".

5. VS Code will build the Docker container and set up the development environment. This may take a few minutes the first time.

6. Once the container is built, you'll have a fully configured environment with all the necessary dependencies installed.

Using Dev Containers ensures that all course participants have the same development environment, regardless of their local setup. It also makes it easier to manage dependencies and avoid conflicts with other projects.


## Teaching Approach

The course will use a **top-down** teaching method, which is different from the traditional **bottom-up** approach. 

- **Top-Down Method**: We'll start with a high-level overview and practical application, then go deep into the underlying details as needed. This approach helps maintain motivation and provides a clearer picture of how different components fit together.
- **Bottom-Up Method**: Typically involves learning individual components in isolation before combining them into more complex structures, which can sometimes lead to a fragmented understanding.

### Example: Learning Baseball
Harvard Professor David Perkins, in his book [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719), compares learning to playing baseball. Kids don't start by memorizing all the rules and technical details; they begin by playing the game and gradually learn the intricacies. Similarly, in this course, you'll start with practical applications and slowly uncover the theoretical aspects.

> **Important**: Don't worry if you don't understand everything initially. Focus on what things do, not what they are. 

### Learning Methods
1. **Doing**: Engage in coding and building projects.
2. **Explaining**: Write about what you've learned or help others in the course.

You'll be encouraged to follow along with coding exercises and explain your learning to others. Summarizing key points as the course progresses will also be part of the learning process.

## Contributing

Contributions to the course repository are welcome! If you have any improvements, bug fixes, or additional resources to share, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/YourFeature`).
6. Create a Pull Request.

## Contact

You can [contact me](elias.jacob@ufrn.br) for any questions or feedback regarding the course materials or repository.
