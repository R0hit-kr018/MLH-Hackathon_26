"""
End-to-end verification script for OpenResearch AI research synthesis pipeline.
"""

from app import get_demo_evidence, synthesize_fallback_report, check_ollama_status

def run_tests():
    print("1. Testing Ollama Status Check...")
    is_up, models = check_ollama_status()
    print(f"   Ollama active: {is_up}, Installed models: {models}")

    print("\n2. Testing Evidence Collection...")
    topic = "What are the latest breakthroughs in solid-state batteries in 2025-2026?"
    evidence = get_demo_evidence(topic)
    assert len(evidence) == 5, f"Expected 5 evidence items, got {len(evidence)}"
    for i, e in enumerate(evidence):
        print(f"   [Source {i+1}] {e['title'][:40]}... (Score: {e['score']})")

    print("\n3. Testing Report Synthesis & Grounding...")
    report = synthesize_fallback_report(topic, evidence, "Gemma 3 4B")
    
    # Verify the 6 mandatory sections
    required_sections = [
        "## 1. Executive Summary",
        "## 2. Key Findings",
        "## 3. Comparison of Different Sources",
        "## 4. Agreements and Conflicting Claims",
        "## 5. Evidence-Based Recommendations",
        "## 6. Limitations and Conclusion",
    ]
    for sec in required_sections:
        assert sec in report, f"Missing required section: {sec}"
        print(f"   Section verified: {sec}")

    # Verify numbered citations
    for i in range(1, 6):
        assert f"[{i}]" in report, f"Missing citation [{i}]"
    print("   All citations [1] to [5] verified in synthesized report.")

    print("\n4. Testing Markdown Export...")
    with open("sample_research_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("   sample_research_report.md generated successfully.")

    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
