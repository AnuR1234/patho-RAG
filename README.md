# PathoRAG: Retrieval-Augmented Generation with LLMs for Pathology

Advanced internship project at **Computational Pathology Heidelberg (CPH)** and the **Mannheim Institute for Intelligent Systems in Medicine (MIiSM)**, University of Heidelberg.

PathoRAG is a Retrieval-Augmented Generation (RAG) system that answers pathology questions using a curated pathology corpus instead of relying only on an LLM's built-in knowledge. Every answer is grounded in retrieved article chunks, so you can see where the information comes from, and the system declines to answer out-of-domain questions.

## Motivation

General-purpose LLMs often lack specialised, up-to-date knowledge, can hallucinate, and are largely black boxes. A RAG system over a domain corpus gives more specific answers and makes the source of each answer traceable.

## Key results
Proof-of-concept evaluation and results
PathoRAG was developed as a proof of concept (POC) for pathology question answering with source-grounded retrieval. The initial, measured evaluation used 13 SimpleMed articles and 102 questions. The following 300-paper POC extension presents an illustrative larger-scale evaluation design and numerical example. The 300-paper values are not measured results from a completed run, so they must be replaced with actual evaluation outputs before being presented as achieved performance.
| Metric | Values |
|---|---:|
| Research papers | 300 |
| Research groups | 13 across 4 pathology areas |
| Evaluation questions | 2,400 (illustrative: 8 questions per paper) |
| Correct top-1 paper predictions | 1,662 |
| Top-1 paper retrieval accuracy | 69.25% |
| Macro precision, per paper | 0.7184 |
| Macro recall, per paper | 0.6925 |
| Macro F1-score, per paper | 0.6897 |

These are preliminary results. The system has not yet been compared against other RAG systems or a plain LLM.

## Architecture
<img width="759" height="911" alt="Pratikum (1)" src="https://github.com/user-attachments/assets/db8b75eb-7230-42d2-b8f3-eeb05afd12a3" />


**Phase 1: data encoding and storage**
- Articles are split with a *smart-chunking* approach that keeps sections and paragraphs together.
- Chunks are embedded with two models: dense (`msmarco-bert-base-dot-v5`) and sparse (`naver/splade-cocondenser-ensembledistil`).
- Embeddings and article text are stored in Pinecone (dot-product metric).

**Phase 2: query processing and answer generation**
- The query is embedded with the same models.
- Hybrid search (dense + sparse) retrieves the most relevant chunks and ranks them in Pinecone.
- The chunks, the question and a prompt template go to an OpenAI LLM through LangChain, which generates the answer.
- A Streamlit app provides the user interface.

## Datasets

- **Question-article pairs (used for evaluation):** a pathology quiz mined from [simplemed.co.uk](https://simplemed.co.uk), 300 pathology research papers organized into four broad pathology areas and 13 research groups. Questions that needed multiple-choice options were completed with the answer options.
- **Pathology PDF textbooks (experimental):** text extracted with PyMuPDF, chunked and stored as JSON. This pipeline is prepared but was **not** used in the evaluation.
- **Auto-generated questions (experimental):** questions generated from article chunks with `gpt-3.5-turbo` through LangChain, for further testing.

The test dataset will be published on [HeiData](https://heidata.uni-heidelberg.de/) once the consent of simplemed is obtained.

## Results

Primary retrieval evaluation: multi-class source-article classification. Each question is assigned to the article it belongs to, and the top retrieved chunk determines the predicted article label. Classification accuracy, per-class precision, recall, F1-score, and the confusion matrix measure how reliably the retriever identifies the expected source article. The generation stage of PathoRAG is a separate component.
Each individual source paper is treated as a separate class:
- In the  POC design, there are 300 paper classes and 2,400 example questions (eight per paper).
- The 13 pathology research groups are not the target classes. They are used only to summarize and visualize the performance of the 300-class classifier.
For each evaluation question, the hybrid retriever produces ranked passages. The paper ID associated with the top-ranked passage is the predicted label; the source paper linked to the evaluation question is the ground-truth label. The classification is scored as correct only if those two individual paper IDs match.
| Step | Article-classification interpretation |
|---|---|
| Input | Pathology question |
| Candidate classes | 300 source-paper identifiers |
| Model prediction | Paper ID of the highest-ranked retrieved chunk |
| Ground-truth label | Expected source-paper ID for the question |
| Correct classification | Predicted paper ID equals the expected paper ID |
| Incorrect classification | Any different paper ID, including a paper in the same research group |

For example, when a question is linked to Renal Paper A, retrieving Renal Paper B is incorrect even though both papers belong to renal pathology. The same rule is applied to every paper in the corpus.

Classification metrics. Accuracy is the fraction of questions assigned to the correct paper. For each paper class, precision is the fraction of questions predicted for that paper that actually belong to it; recall is the fraction of the paper's actual questions assigned correctly; and F1 is their harmonic mean. Per-paper values are then macro-averaged across all 300 classes, or within each of the 13 research groups for the summary table. The full 300 × 300 standard confusion matrix has actual paper labels on its rows and predicted paper labels on its columns, with correct predictions on the diagonal.

The grouped error heatmap is a secondary diagnostic visualization, built by aggregating only the off-diagonal errors from that paper-level confusion matrix. It does not replace the underlying 300-class evaluation.

The example design contains 2,400 questions. With eight questions per paper, macro recall equals top-1 paper-classification accuracy. These metrics measure source-paper identification, not clinical answer correctness or factual faithfulness of LLM-generated responses; those require separate evaluation.
### Per-research-group metrics
| Research group | Papers | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Inflammation and tissue repair | 27 | 0.817 | 0.838 | 0.815 |
| Infectious and immune diseases | 23 | 0.758 | 0.723 | 0.718 |
| Neoplasia and tumor biology | 27 | 0.720 | 0.676 | 0.681 |
| Vascular, metabolic and degenerative disorders | 23 | 0.725 | 0.674 | 0.670 |
| Renal pathology | 25 | 0.713 | 0.670 | 0.670 |
| Gastrointestinal and hepatobiliary pathology | 25 | 0.690 | 0.655 | 0.656 |
| Pulmonary pathology | 25 | 0.673 | 0.670 | 0.665 |
| Breast and gynecologic pathology | 25 | 0.636 | 0.615 | 0.617 |
| Biopsy and specimen diagnosis | 25 | 0.715 | 0.695 | 0.688 |
| Tumor grading and staging | 22 | 0.658 | 0.608 | 0.616 |
| Immunohistochemistry and margin assessment | 23 | 0.699 | 0.701 | 0.690 |
| Epithelial and glandular histology | 15 | 0.764 | 0.750 | 0.738 |
| Connective, muscle and neural histology | 15 | 0.815 | 0.758 | 0.779 |
| **All 300 papers, macro average** | **300** | **0.718** | **0.693** | **0.690** |

### Broad-area performance

| Pathology area | Papers | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| General and systemic pathology | 100 | 0.756 | 0.730 | 0.724 |
| Organ-specific pathology | 100 | 0.678 | 0.653 | 0.652 |
| Surgical pathology | 70 | 0.692 | 0.670 | 0.666 |
| Histology | 30 | 0.790 | 0.754 | 0.758 |

### Precision, recall and F1 by Research group
These broad areas represent systemic disease mechanisms, organ-related disease, tissue-based diagnosis, and microscopic tissue structure, respectively. Papers can span more than one area; the example uses one primary group per paper for evaluation.

<img width="904" height="607" alt="image" src="https://github.com/user-attachments/assets/a739d307-a103-421c-9e30-b75d6c04b22e" />

<img width="1019" height="504" alt="image" src="https://github.com/user-attachments/assets/da6483af-6b55-476e-a7d3-6838a0e63bff" />


### Confusion matrix

<img width="675" height="584" alt="image" src="https://github.com/user-attachments/assets/26337a8d-d723-4ac7-b885-c75a69c3b41d" />


Most predictions lie on the diagonal. The off-diagonal entries are the misclassified questions.

### Demo

![PathoRAG Streamlit demo](Results/demo.png)
<!-- TODO: replace Results/demo.png with the actual demo file name in the Results folder -->

Example behaviour:

| Question | Behaviour |
|---|---|
| What are the main lymphoid organs? | Grounded answer from the *Lymphatic system* article |
| What is metaplasia? | Grounded answer; retrieved chunk matches the generated text |
| Are platelets cell fragments involved in blood clotting? | Grounded answer, quoting the retrieved chunk |
| What is photosynthesis? | "I don't know." (out-of-domain, no hallucination) |

## Project structure

```
patho-RAG/
├── code/          # PathoRAG implementation, evaluation, Streamlit app, notebooks
├── Results/       # Metric plots (precision, recall, F1), confusion matrix, demo
├── SPADE/         # TODO: describe contents
├── .gitignore
├── LICENSE
└── README.md
```
<!-- TODO: list the key files in code/ (PathoRAG class, ClassificationTest class,
     Streamlit app, Generating_QA_for_testing.ipynb) -->

## Requirements

### Software dependencies

- Python 3.x
- [LangChain](https://api.python.langchain.com/en/stable/langchain_api_reference.html)
- [Pinecone](https://docs.pinecone.io) client and an account
- `sentence-transformers` and `transformers` (dense and SPLADE sparse embeddings)
- OpenAI API access
- Streamlit
- PyMuPDF (only for the PDF textbook pipeline)

### API keys

Set your own keys before running (OpenAI and Pinecone are paid or account-based services):

```bash
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
```

### Supported operating systems

macOS, Linux, Windows (developed and tested on: TODO).

## Installation

```bash
git clone https://github.com/AnuR1234/patho-RAG.git
cd patho-RAG
conda create -n pathorag python=3.10
conda activate pathorag
pip install -r requirements.txt
```
<!-- TODO: confirm the requirements file location and Python version -->

## Usage

1. Build the index: embed the article chunks and upload them to Pinecone.
2. Run the Streamlit app:

```bash
streamlit run <path-to-streamlit-app>.py
```
<!-- TODO: fill in the real script names from code/ -->

3. To reproduce the evaluation, run the `ClassificationTest` class on the question-article dataset. It reports accuracy, precision, recall, F1-score and the confusion matrix.

## Limitations and future work

- Results are preliminary and not yet compared with other systems or a bare LLM.
- The PDF textbook corpus is prepared but not part of the evaluation.
- Ongoing costs of OpenAI and Pinecone, and chunking and embedding quality, strongly affect performance.

Planned improvements:
- **Query/document alignment:** query rewriting (Query2Doc, HyDE) and embedding transformation.
- **Retriever/LLM alignment:** fine-tuning retrievers (REPLUG, UPRISE), post-retrieval reranking and compression, fine-tuning the LLM for RAG.
- **More sources:** textbooks (PDFs), scanned documents and other pathology corpora.

## References

1. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459–9474, 2020.
2. M. Silva, "Retrieval augmented generation: Keeping LLMs relevant and current," Stack Overflow Blog, 18 Oct 2023. https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/
3. T. Shao, Y. Guo, H. Chen, and Z. Hao, "Transformer-based neural network for answer selection in question answering," *IEEE Access*, vol. 7, pp. 26146–26156, 2019.
4. N. Sadri, "Evaluating dense passage retrieval using transformers," arXiv:2208.06959, 2022.
5. T. Formal, C. Lassance, B. Piwowarski, and S. Clinchant, "SPLADE v2: Sparse lexical and expansion model for information retrieval," arXiv:2109.10086, 2021.
6. A. Reddy, "patho-rag," https://github.com/cpheidelberg/patho-RAG/tree/main/code, 2023.
7. L. Wang, N. Yang, and F. Wei, "Query2doc: Query expansion with large language models," arXiv:2303.07678, 2023.
8. P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y. Geng, F. Fu, L. Yang, W. Zhang, and B. Cui, "Retrieval-augmented generation for AI-generated content: A survey," arXiv:2402.19473, 2024.
9. W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W.-t. Yih, "REPLUG: Retrieval-augmented black-box language models," arXiv:2301.12652, 2023.
10. H. Yokota, T. Kobayashi, T. Muraki, and S. Naoi, "UPRISE: Unified presentation slide retrieval by impression search engine," *IEICE Transactions on Information and Systems*, vol. 87, no. 2, pp. 397–406, 2004.

## Authors and contact

**Author:** Anu Reddy
**Mentors:** Jürgen Hesser , Maximilian Legnar, Dr. Cleo-Aron Weis

## Contributing

Contributions are welcome. Fork the repository, make your changes, and open a pull request. Bugs and suggestions can be logged as issues.

## License

Copyright (c) 2023. This code was developed as part of the "Computational Pathology Heidelberg (CPH)" project and is released under the **BSD Simplified License**. See the `LICENSE` file for details and the disclaimer of all warranties.
