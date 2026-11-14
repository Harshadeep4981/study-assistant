"""
End-to-End Integration, Performance, and Security Benchmark Script.

Executes real and mocked flows, performance profiling, concurrency testing,
and security checks against the active backend without modifying application code.
Masks API keys in all outputs.
"""

import asyncio
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

import fitz
import httpx

# Ensure backend directory is in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.config import get_settings
from app.main import create_app
from app.models.generation import FlashcardGenerateResponse, QuizGenerateResponse


def mask_key(key: str | None) -> str:
    """Mask API key for safe reporting."""
    if not key:
        return "None"
    return f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "***"


def generate_sample_biology_pdf() -> bytes:
    """Generate a realistic 2-page biology study notes PDF for E2E testing."""
    doc = fitz.open()

    page1 = doc.new_page()
    page1_text = (
        "Cellular Biology and Photosynthesis Study Guide\n\n"
        "1. Overview of Photosynthesis\n"
        "Photosynthesis is the biochemical process by which photoautotrophic organisms, "
        "primarily green plants and cyanobacteria, convert solar radiant energy into "
        "stored chemical energy in the form of hexose sugars (glucose, C6H12O6).\n\n"
        "2. The Chemical Equation\n"
        "The overall balanced equation for oxygenic photosynthesis is:\n"
        "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2\n"
        "This reaction requires chlorophyll pigments located within the thylakoid membranes "
        "of chloroplasts.\n\n"
        "3. Light-Dependent Reactions\n"
        "Light reactions occur on thylakoid membranes where photon absorption excites electrons "
        "in Photosystem II (P680) and Photosystem I (P700), generating ATP and NADPH via "
        "photophosphorylation while splitting water molecules (photolysis) to release oxygen gas."
    )
    page1.insert_text((50, 72), page1_text, fontsize=11)

    page2 = doc.new_page()
    page2_text = (
        "4. The Calvin Cycle (Light-Independent Reactions)\n"
        "The Calvin Cycle takes place in the stroma of the chloroplast. It fixes atmospheric "
        "carbon dioxide into organic triose phosphates using ATP and NADPH generated during "
        "light-dependent reactions. The key regulatory enzyme catalyzing the initial carboxylation "
        "step is Ribulose-1,5-bisphosphate carboxylase-oxygenase, commonly known as RuBisCO.\n\n"
        "5. Cellular Respiration Comparison\n"
        "Unlike photosynthesis, cellular respiration is a catabolic process occurring primarily "
        "in eukaryotic mitochondria. It breaks down glucose into carbon dioxide and water, releasing "
        "approximately 30 to 32 ATP molecules per glucose molecule under aerobic conditions through "
        "glycolysis, the citric acid cycle (Krebs cycle), and oxidative phosphorylation."
    )
    page2.insert_text((50, 72), page2_text, fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


async def run_benchmarks():
    print("=" * 70)
    print("AI STUDY ASSISTANT: CRITICAL INTEGRATION & BENCHMARK SUITE")
    print("=" * 70)

    tracemalloc.start()
    settings = get_settings()
    app = create_app(settings)

    print(f"Backend Directory: {_BACKEND_DIR}")
    print(f"Configured Model: {settings.GROQ_MODEL}")
    print(f"Groq API Key: {mask_key(settings.GROQ_API_KEY)}")
    print(f"Max Upload Size: {settings.MAX_UPLOAD_SIZE_BYTES} bytes")
    print(f"Max Generation Input: {settings.MAX_GENERATION_INPUT_BYTES} bytes")
    print(f"Derived Body Limit: {settings.MAX_GENERATION_REQUEST_BODY_BYTES} bytes")
    print("-" * 70)

    # Connect to the live running server on localhost:8000
    base_url = "http://127.0.0.1:8000"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # -------------------------------------------------------------
        # Benchmark 1: Health Check
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        health_resp = await client.get("/api/health")
        t_health = (time.perf_counter() - t0) * 1000
        print(f"\n[1] Health Check: Status {health_resp.status_code} ({t_health:.2f} ms)")
        print(f"    Payload: {health_resp.json()}")
        assert health_resp.status_code == 200

        # -------------------------------------------------------------
        # Benchmark 2: Document Ingestion Performance
        # -------------------------------------------------------------
        print("\n[2] Document Ingestion Latency & Extraction Benchmarks:")

        # 2a: Pasted Text (1 KB, 10 KB, 50 KB)
        for size_kb in [1, 10, 50]:
            raw_text = "Photosynthesis creates glucose. " * (size_kb * 35)
            raw_bytes = len(raw_text.encode("utf-8"))
            t0 = time.perf_counter()
            resp = await client.post("/api/documents/ingest", json={"text": raw_text})
            dt = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200
            print(f"    - Pasted Text ({raw_bytes:,} bytes): {dt:.2f} ms (Status: {resp.status_code})")

        # 2b: PDF Extraction (1 page, 5 pages, 15 pages)
        for pages in [1, 5, 15]:
            pdf_bytes = fitz.open()
            for p in range(pages):
                pg = pdf_bytes.new_page()
                pg.insert_text((50, 72), f"Biology page {p+1}. Cellular ATP production and enzyme mechanisms.")
            data = pdf_bytes.tobytes()
            pdf_bytes.close()

            t0 = time.perf_counter()
            resp = await client.post(
                "/api/documents/ingest",
                files={"file": (f"test_{pages}p.pdf", data, "application/pdf")},
            )
            dt = (time.perf_counter() - t0) * 1000
            assert resp.status_code == 200
            meta = resp.json()["metadata"]
            print(
                f"    - PDF ({pages} pages, {len(data):,} bytes): {dt:.2f} ms "
                f"(Words: {meta['word_count']}, Chars: {meta['character_count']})"
            )

        # -------------------------------------------------------------
        # Benchmark 3: Genuine End-to-End Pipeline with Real Groq AI
        # -------------------------------------------------------------
        print("\n[3] Genuine End-to-End Pipeline (Real Groq Provider):")
        sample_pdf_bytes = generate_sample_biology_pdf()

        # Step 3.1: Ingest sample biology PDF
        t0 = time.perf_counter()
        ingest_resp = await client.post(
            "/api/documents/ingest",
            files={"file": ("Cellular_Biology_Guide.pdf", sample_pdf_bytes, "application/pdf")},
        )
        t_ingest = (time.perf_counter() - t0) * 1000
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        study_text = ingest_data["extracted_text"]
        print(f"    3.1 Ingestion & Extraction: {t_ingest:.2f} ms")
        print(f"        Filename: {ingest_data['metadata']['filename']}")
        print(f"        Pages: {ingest_data['metadata']['page_count']}")
        print(f"        Word Count: {ingest_data['metadata']['word_count']}")
        print(f"        Has Images: {ingest_data['metadata']['has_images']}")

        # Step 3.2: Generate Flashcards (Real Groq)
        t0 = time.perf_counter()
        fc_resp = await client.post(
            "/api/generate/flashcards",
            json={"text": study_text, "count": 3},
        )
        t_fc = (time.perf_counter() - t0) * 1000
        print(f"    3.2 Flashcards Generation (count=3): {t_fc:.2f} ms (Status: {fc_resp.status_code})")
        assert fc_resp.status_code == 200
        fc_model = FlashcardGenerateResponse.model_validate(fc_resp.json())
        assert fc_model.count == 3
        assert len(fc_model.flashcards) == 3
        for i, fc in enumerate(fc_model.flashcards, 1):
            print(f"        Flashcard #{i}:")
            print(f"          Q: {fc.question}")
            print(f"          A: {fc.answer}")
            assert len(fc.question.strip()) > 5
            assert len(fc.answer.strip()) > 3

        # Step 3.3: Generate Quiz (Real Groq)
        t0 = time.perf_counter()
        quiz_resp = await client.post(
            "/api/generate/quiz",
            json={"text": study_text, "count": 2},
        )
        t_quiz = (time.perf_counter() - t0) * 1000
        print(f"    3.3 Quiz Generation (count=2): {t_quiz:.2f} ms (Status: {quiz_resp.status_code})")
        assert quiz_resp.status_code == 200
        quiz_model = QuizGenerateResponse.model_validate(quiz_resp.json())
        assert quiz_model.count == 2
        assert len(quiz_model.quiz) == 2
        for i, q in enumerate(quiz_model.quiz, 1):
            print(f"        Quiz #{i}:")
            print(f"          Q: {q.question}")
            print(f"          Options: {q.options}")
            print(f"          Correct Answer: {q.correct_answer}")
            print(f"          Explanation: {q.explanation}")
            # Cross-field validations
            assert len(q.options) == 4
            assert len(set(q.options)) == 4, "Duplicate choices detected!"
            assert q.correct_answer in q.options, "correct_answer not in options!"
            assert len(q.explanation.strip()) > 5, "Empty explanation!"

        # Step 3.4: Verify Flow Handles Failure Without Crashing
        bad_fc_resp = await client.post(
            "/api/generate/flashcards",
            json={"text": study_text, "count": 50},  # Exceeds max count of 30
        )
        print(f"    3.4 Pipeline Failure Handling (count=50): Status {bad_fc_resp.status_code} (Clean 422)")
        assert bad_fc_resp.status_code == 422

        # -------------------------------------------------------------
        # Benchmark 4: Concurrency & Isolation
        # -------------------------------------------------------------
        print("\n[4] Concurrency Benchmark (5 Parallel Requests):")
        concurrent_payloads = [
            {"text": f"Study topic #{i}: ATP generation in mitochondria during aerobic respiration.", "count": 1}
            for i in range(1, 6)
        ]

        t0 = time.perf_counter()
        tasks = [client.post("/api/generate/flashcards", json=p) for p in concurrent_payloads]
        concur_responses = await asyncio.gather(*tasks)
        t_concur = (time.perf_counter() - t0) * 1000
        avg_latency = t_concur / len(tasks)

        all_ok = all(r.status_code == 200 for r in concur_responses)
        print(f"    Total Time for 5 Concurrent Requests: {t_concur:.2f} ms (Avg: {avg_latency:.2f} ms/req)")
        print(f"    All 5 Succeeded (200 OK): {all_ok}")
        assert all_ok

        # -------------------------------------------------------------
        # Benchmark 5: Security & Robustness Verification
        # -------------------------------------------------------------
        print("\n[5] Security & Robustness Checks:")

        # 5a: API Key leakage in response bodies and headers
        raw_health = health_resp.text
        raw_fc = fc_resp.text
        raw_quiz = quiz_resp.text
        key = settings.GROQ_API_KEY
        leaked = False
        if key and (key in raw_health or key in raw_fc or key in raw_quiz):
            leaked = True
        print(f"    - API Key Leaked in Responses: {leaked} (PASS: Key is never exposed)")
        assert not leaked

        # 5b: CORS Enforcement
        cors_req = await client.options(
            "/api/generate/flashcards",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
        print(f"    - CORS Preflight (allowed http://localhost:3000): Status {cors_req.status_code}")
        assert cors_req.headers.get("access-control-allow-origin") == "http://localhost:3000"

        disallowed_cors = await client.options(
            "/api/generate/flashcards",
            headers={"Origin": "https://malicious-site.com", "Access-Control-Request-Method": "POST"},
        )
        print(f"    - CORS Preflight (disallowed origin): Header present: {'access-control-allow-origin' in disallowed_cors.headers}")
        assert "access-control-allow-origin" not in disallowed_cors.headers

        # 5c: Upload Size Limits
        oversized_json = "X" * 150000
        oversized_resp = await client.post("/api/generate/flashcards", json={"text": oversized_json, "count": 1})
        print(f"    - Oversized Text (150 KB > 100 KB limit): Status {oversized_resp.status_code} ({oversized_resp.json().get('detail')})")
        assert oversized_resp.status_code == 413

        # 5d: Memory profiling
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"\n[6] Memory Profile:")
        print(f"    - Current Traced Memory: {current_mem / 1024 / 1024:.2f} MB")
        print(f"    - Peak Traced Memory:    {peak_mem / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY (100% PASS)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
