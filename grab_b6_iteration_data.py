import glob
import jsonlines
import pandas as pd


def main() -> None:
    files = glob.glob("results/hf_s1_results_*_B6_*reextract*jsonl")
    other_files = glob.glob("results/mirage_s1_results_*_B6_*reextract*jsonl")

    all_files = files + other_files

    # remove files with 'eval' in the name
    all_files = [f for f in all_files if "eval" not in f]

    rows = []

    # go through all files and collect iteration data
    for file in all_files:
        dataset = "MIRAGE"
        if "pubmedqa" in file:
            dataset = "PubMedQA"
        elif "medical" in file:
            dataset = "Medical exam"

        model = "Gemma-3 4B"
        if "gemma-3-1b" in file:
            model = "Gemma-3 1B"
        elif "gemma-3-27b" in file:
            model = "Gemma-3 27B"
        elif "medgemma-4b" in file:
            model = "MedGemma 4B"
        elif "medgemma-27b" in file:
            model = "MedGemma 27B"

        token_budget = 128
        if "256words" in file:
            token_budget = 256
        elif "512words" in file:
            token_budget = 512
        elif "768words" in file:
            token_budget = 768
        elif "1024words" in file:
            token_budget = 1024

        data = list(jsonlines.open(file, "r"))
        all_iterations = [packet["llm_response"]["total_iterations"] for packet in data]
        if dataset == "MIRAGE":
            pubmedqa_no_context_iterations = [packet["llm_response"]["total_iterations"] for packet in data if "pubmedqa" in packet["dataset"]]

        for it in all_iterations:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "token_budget": token_budget,
                    "total_iterations": it,
                    "source_file": file,
                }
            )
        
        if dataset == "MIRAGE" and len(pubmedqa_no_context_iterations):
            for it in pubmedqa_no_context_iterations:
                rows.append(
                    {
                        "dataset": "PubMedQA (no context)",
                        "model": model,
                        "token_budget": token_budget,
                        "total_iterations": it,
                        "source_file": file,
                    }
                )

    if not rows:
        print("No matching B6 re-extract result files found.")
        return

    df = pd.DataFrame(rows)
    output_path = "b6_iterations_table.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved B6 iteration data to {output_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()
