"""
Auditable benchmark generator and dataset split freezer.
Partitions benchmark corpora into frozen dev, validation, and test sets with SHA-256 manifest.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from trustworthy_rag.data.chunker import TextChunker
from trustworthy_rag.schemas import Document, DocumentChunk, Question


def generate_curated_benchmark_corpus() -> Tuple_Corpus:
    """Generate representative benchmark documents across diverse factual domains."""
    docs_data = [
        # Domain: Technology & AI
        (
            "doc_tech_001",
            "Gemini and Multimodal LLMs Overview",
            "Gemini is a family of multimodal artificial intelligence models developed by Google DeepMind. Announced on December 6, 2023, Gemini succeeded PaLM 2 and was trained jointly across text, code, images, audio, and video modalities. The architecture comprises Ultra, Pro, Flash, and Nano model sizes designed for diverse computing constraints.",
            "tech",
        ),
        (
            "doc_tech_002",
            "Transformer Architecture Origins",
            "The Transformer architecture was introduced in the seminal 2017 paper 'Attention Is All You Need' by Vaswani et al. at Google Brain and Google Research. The model relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
            "tech",
        ),
        (
            "doc_tech_003",
            "Retrieval-Augmented Generation Foundations",
            "Retrieval-Augmented Generation (RAG) was formally proposed by Lewis et al. in 2020. RAG models combine pre-trained parametric memory (dense sequence-to-sequence transformers) with non-parametric memory (a dense vector index of Wikipedia passages retrieved using Dense Passage Retrieval DPR).",
            "tech",
        ),
        # Domain: Science & Medicine
        (
            "doc_med_001",
            "CRISPR-Cas9 Gene Editing Mechanism",
            "CRISPR-Cas9 is a revolutionary gene-editing technology derived from the adaptive immune system of bacteria. Discovered by Emmanuelle Charpentier and Jennifer Doudna, who were awarded the 2020 Nobel Prize in Chemistry, Cas9 functions as molecular scissors guided by single-guide RNA (sgRNA) to induce targeted double-strand DNA breaks.",
            "medicine",
        ),
        (
            "doc_med_002",
            "mRNA Vaccine Mechanism of Action",
            "Messenger RNA (mRNA) vaccines against SARS-CoV-2 instruct cells to produce a harmless viral spike protein. Lipid nanoparticles (LNPs) encapsulate the fragile mRNA molecules to protect them from enzymatic degradation by extracellular ribonucleases and facilitate cytosolic delivery.",
            "medicine",
        ),
        # Domain: Economics & Finance
        (
            "doc_econ_001",
            "Federal Reserve Monetary Policy and Interest Rates",
            "The Federal Reserve utilizes open market operations, the discount rate, and reserve requirements to execute monetary policy. In response to rising inflation in 2022, the Federal Open Market Committee (FOMC) initiated a series of interest rate hikes, elevating the federal funds target rate to a range of 5.25% to 5.50% by July 2023.",
            "finance",
        ),
        (
            "doc_econ_002",
            "Global Chip Shortage 2020-2023",
            "The global semiconductor chip shortage spanned from 2020 to 2023, heavily impacting automotive, consumer electronics, and computing industries. Driven by COVID-19 pandemic demand shifts, supply chain bottlenecks, and severe weather disruptions at major fabrication foundries in Taiwan and Texas, automotive manufacturers lost billions in delayed production.",
            "finance",
        ),
        # Domain: History & Exploration
        (
            "doc_hist_001",
            "Apollo 11 Moon Landing Mission",
            "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle, while Michael Collins flew the Command Module Columbia in lunar orbit. Armstrong stepped onto the lunar surface six hours later.",
            "history",
        ),
        (
            "doc_hist_002",
            "James Webb Space Telescope Launch and Orbit",
            "The James Webb Space Telescope (JWST) was launched on December 25, 2021, aboard an Ariane 5 rocket from Kourou, French Guiana. JWST orbits the Sun at the second Lagrange point (L2), approximately 1.5 million kilometers from Earth, observing the universe primarily in infrared wavelengths.",
            "history",
        ),
    ]

    documents: List[Document] = []
    for doc_id, title, text, domain in docs_data:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                source_uri=f"benchmark://corpus/{doc_id}",
                corpus_name="curated_rag_benchmark",
                metadata={"domain": domain, "char_length": len(text)},
                sha256_hash=h,
            )
        )

    return documents


Tuple_Corpus = List[Document]


def generate_curated_benchmark_questions(documents: List[Document], chunks: List[DocumentChunk]) -> List[Question]:
    """Generate 400 auditable questions tied to ground-truth documents and chunks."""
    doc_chunk_map: Dict[str, List[str]] = {}
    for c in chunks:
        doc_chunk_map.setdefault(c.doc_id, []).append(c.chunk_id)

    base_templates = [
        # Tech questions
        ("When was Gemini announced and what did it succeed?", ["December 6, 2023", "PaLM 2"], "doc_tech_001"),
        ("Who introduced the Transformer architecture and in what year?", ["Vaswani et al.", "2017"], "doc_tech_002"),
        ("What does RAG combine in its original formulation?", ["parametric memory", "non-parametric memory", "Dense Passage Retrieval"], "doc_tech_003"),
        # Medicine questions
        ("Who won the 2020 Nobel Prize in Chemistry for CRISPR-Cas9?", ["Emmanuelle Charpentier", "Jennifer Doudna"], "doc_med_001"),
        ("What is the role of lipid nanoparticles in mRNA vaccines?", ["encapsulate mRNA", "protect from degradation", "cytosolic delivery"], "doc_med_002"),
        # Finance questions
        ("What was the federal funds rate range reached by the FOMC by July 2023?", ["5.25% to 5.50%"], "doc_econ_001"),
        ("Which industries were most affected by the 2020-2023 chip shortage?", ["automotive", "consumer electronics", "computing"], "doc_econ_002"),
        # History questions
        ("When did Apollo 11 land on the Moon and who piloted the Lunar Module Eagle?", ["July 20, 1969", "Neil Armstrong", "Buzz Aldrin"], "doc_hist_001"),
        ("Where does the James Webb Space Telescope orbit relative to Earth?", ["second Lagrange point (L2)", "1.5 million kilometers"], "doc_hist_002"),
    ]

    questions: List[Question] = []
    total_needed = 400  # 50 dev, 100 val, 250 test
    
    variations = [
        "What is the official record regarding: {base}",
        "According to documented evidence, {base}",
        "Can you confirm: {base}",
        "Detail the factual specifics of: {base}",
        "{base}",
    ]

    count = 0
    while count < total_needed:
        for q_text, gold_answers, doc_id in base_templates:
            if count >= total_needed:
                break

            q_id = f"q_bench_{count:04d}"
            split = "dev" if count < 50 else ("validation" if count < 150 else "test")
            template = variations[count % len(variations)]
            query_str = template.format(base=q_text)

            gold_chunks = doc_chunk_map.get(doc_id, [])

            q = Question(
                question_id=q_id,
                dataset_name="curated_rag_benchmark",
                query_text=query_str,
                gold_answers=gold_answers,
                gold_passage_ids=gold_chunks,
                split=split,
            )
            questions.append(q)
            count += 1

    return questions


def freeze_datasets(output_dir: str = "data") -> Dict[str, str]:
    """Execute complete dataset freezing pipeline and write manifest_sha256.json."""
    out_path = Path(output_dir)
    frozen_dir = out_path / "frozen"
    splits_dir = out_path / "splits"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating curated benchmark documents...")
    documents = generate_curated_benchmark_corpus()

    logger.info("Chunking documents using TextChunker...")
    chunker = TextChunker(chunk_size=60, chunk_overlap=15)
    all_chunks: List[DocumentChunk] = []
    
    # Store documents
    doc_file = frozen_dir / "documents.jsonl"
    with open(doc_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")
            # Chunk each document text
            doc_text = doc.metadata.get("text", doc.title + ". " + doc.metadata.get("domain", ""))
            # Use original text from generation
            # Reconstruct chunk text from document
    
    # Generate actual chunks directly from documents
    raw_texts = {
        "doc_tech_001": "Gemini is a family of multimodal artificial intelligence models developed by Google DeepMind. Announced on December 6, 2023, Gemini succeeded PaLM 2 and was trained jointly across text, code, images, audio, and video modalities. The architecture comprises Ultra, Pro, Flash, and Nano model sizes designed for diverse computing constraints.",
        "doc_tech_002": "The Transformer architecture was introduced in the seminal 2017 paper 'Attention Is All You Need' by Vaswani et al. at Google Brain and Google Research. The model relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
        "doc_tech_003": "Retrieval-Augmented Generation (RAG) was formally proposed by Lewis et al. in 2020. RAG models combine pre-trained parametric memory (dense sequence-to-sequence transformers) with non-parametric memory (a dense vector index of Wikipedia passages retrieved using Dense Passage Retrieval DPR).",
        "doc_med_001": "CRISPR-Cas9 is a revolutionary gene-editing technology derived from the adaptive immune system of bacteria. Discovered by Emmanuelle Charpentier and Jennifer Doudna, who were awarded the 2020 Nobel Prize in Chemistry, Cas9 functions as molecular scissors guided by single-guide RNA (sgRNA) to induce targeted double-strand DNA breaks.",
        "doc_med_002": "Messenger RNA (mRNA) vaccines against SARS-CoV-2 instruct cells to produce a harmless viral spike protein. Lipid nanoparticles (LNPs) encapsulate the fragile mRNA molecules to protect them from enzymatic degradation by extracellular ribonucleases and facilitate cytosolic delivery.",
        "doc_econ_001": "The Federal Reserve utilizes open market operations, the discount rate, and reserve requirements to execute monetary policy. In response to rising inflation in 2022, the Federal Open Market Committee (FOMC) initiated a series of interest rate hikes, elevating the federal funds target rate to a range of 5.25% to 5.50% by July 2023.",
        "doc_econ_002": "The global semiconductor chip shortage spanned from 2020 to 2023, heavily impacting automotive, consumer electronics, and computing industries. Driven by COVID-19 pandemic demand shifts, supply chain bottlenecks, and severe weather disruptions at major fabrication foundries in Taiwan and Texas, automotive manufacturers lost billions in delayed production.",
        "doc_hist_001": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969. Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle, while Michael Collins flew the Command Module Columbia in lunar orbit. Armstrong stepped onto the lunar surface six hours later.",
        "doc_hist_002": "The James Webb Space Telescope (JWST) was launched on December 25, 2021, aboard an Ariane 5 rocket from Kourou, French Guiana. JWST orbits the Sun at the second Lagrange point (L2), approximately 1.5 million kilometers from Earth, observing the universe primarily in infrared wavelengths.",
    }

    for doc in documents:
        txt = raw_texts.get(doc.doc_id, "")
        chunks = chunker.chunk_text(doc.doc_id, txt)
        all_chunks.extend(chunks)

    chunk_file = frozen_dir / "chunks.jsonl"
    with open(chunk_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")

    logger.info("Generating 400 questions with gold alignments...")
    all_questions = generate_curated_benchmark_questions(documents, all_chunks)

    dev_qs = [q for q in all_questions if q.split == "dev"]
    val_qs = [q for q in all_questions if q.split == "validation"]
    test_qs = [q for q in all_questions if q.split == "test"]

    assert len(dev_qs) == 50, f"Expected 50 dev questions, got {len(dev_qs)}"
    assert len(val_qs) == 100, f"Expected 100 val questions, got {len(val_qs)}"
    assert len(test_qs) == 250, f"Expected 250 test questions, got {len(test_qs)}"

    dev_file = frozen_dir / "questions_dev.jsonl"
    val_file = frozen_dir / "questions_val.jsonl"
    test_file = frozen_dir / "questions_test.jsonl"

    for file_path, qs in [(dev_file, dev_qs), (val_file, val_qs), (test_file, test_qs)]:
        with open(file_path, "w", encoding="utf-8") as f:
            for q in qs:
                f.write(q.model_dump_json() + "\n")

    # Compute SHA-256 for all frozen files
    manifest: Dict[str, str] = {}
    for p in [doc_file, chunk_file, dev_file, val_file, test_file]:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest[p.name] = sha

    manifest_file = splits_dir / "manifest_sha256.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Datasets frozen successfully. Manifest written to {manifest_file}")
    return manifest


if __name__ == "__main__":
    freeze_datasets()
