# Master’s Internship at CPH (Computation Pathology Heidelberg)
Topic: Retrieval-augmented LLM for pathological use-case


<img width="629" alt="Screenshot 2023-12-09 at 22 01 44" src="https://github.com/cpheidelberg/patho-RAG/assets/35266536/77b6c92e-9c5d-466b-876c-7ac036ccbabe">

Summary (our motivation for taking up this project):
In order to reduce the hallucinations and provide more domain-specific output rather than generalized data. 
Since most of the LLMs are ‘blackbox’, having an RAG system for our owndata, would be clearer as to where the data is been taken from in order to generate new text.

Introduction:
The main two components of a context-document-retriever are, therefore, 
retriever 
generator.
The retriever part can be described as a system that can encode our data so that the relevant parts of it can be easily retrieved upon querying it. The encoding uses text embeddings, i.e., a model trained to create a vector representation of the information. The best option for storing the embeddings of the retriever is a vector database. As a vector database, there are multiple options. A few examples are ChromaDBB, FAISS, Pinecone, and Weaviate. Our option in this will be either PineCone or ChromaDB(yet to decide) 
For the generator part, the obvious option is a LLM. In this workflow, we will be using Medalpeca from the huggingface instance. The orchestration of the retriever and generator will be done using Langchain. 
Target Definition:
Design of the smart document retrieval system that can be fed to LLM(Medalpeca) to provide a more concise response to users' queries.




**Copyright (c) 2023.**

This code has been developed as part of the "Computational Pathology Heidelberg (CPH)" project.

For detailed information on usage, redistribution, and a DISCLAIMER OF ALL WARRANTIES, please refer to the "LICENSE" file in this distribution.

**BSD Simplified License**

## GitHub Repository Naming
To name your repositories, please use the following naming convention:

`[Prefix]-[Project Name]-[Tech Stack]`

- **Prefix**: Use prefixes to categorize repositories, e.g.,
  - `proj-` for project repositories
  - `lib-` for libraries
  - `doc-` for documentation
  - `app-` for applications
  - `data-` for data-related repositories

- **Project Name**: Provide a concise description of the project or repository's main purpose. This should be the most prominent part of the name.

- **Tech Stack**: Specify the technology, framework, tool, etc., on which the project is based.

- **Use CamelCase for Multi-Word Names**: If necessary, use camelCase for multi-word names, e.g., "computationalPathology."

- **No Underscores or Mixed Case**: Stick to hyphens and lowercase letters for simplicity and consistency.

- **No Special Characters**: Avoid special characters, spaces, or symbols in repository names.

For example, if you have a machine learning project related to image processing using Pytorch and OpenCV you can use:

- `proj-image-processing-pytorch-opencv`
- `proj-imageProcessing-pytorch-opencv`

## Description
Detailed project description, including relevant attribution for forked, adapted, or inspired projects.

### Project Structure
For larger, more complicated projects, this is the repo folder structure:

- `folder1/`: Contains models.
- `folder2/`: Contains media assets.
- ...

## Requirements

### Software Dependencies
- libA (version X.Y.Z)
- libB (version A.B.C)
- ...

### Supported Operating Systems
- macOS
- Linux
- Windows
- ...
  
(Add any notes about the platforms or setups used for development & testing.)

## Installation & Build

_General dependency & build install instructions with per-platform subsections when needed. Please include commandline steps, when applicable._

To get started, follow these steps:
1. Download and install [Anaconda](https://www.anaconda.com/products/individual).
2. Create a virtual environment:
```shell
$ conda create -n "name" python=3.7
$ conda activate "name"
$ pip install -r requirements.txt
```
3. Install project dependencies
```shell
$ pip install -r requirements.txt
```
(Include any platform-specific instructions if needed.)

## Usage
After installation, you can use the application as follows:

### Show Help
```
$ python executable.py --help
```

### Scenario 1
```
$ python executable.py --parameter 42
```

(Provide more usage examples if applicable.)

## References / Further Reading
Here are some useful references and related projects:
- [paper](www.google.com)
- [paper2](www.google.com)

## Contribute
Contributions are very welcome! Here's how to get involved:

1. Clone or fork the repository.
2. Make your changes or improvements.
3. Create a pull request.
4. If you find any bugs or have suggestions, please log them here as well.

(Include additional details on development environment setup, coding style, testing, and issue reporting as needed.)

---
**Author:** Your Name <br>
**Contact:** [your@email.com](mailto:your@email.com)
