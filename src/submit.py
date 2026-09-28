import pandas as pd


def save_csv_answer(predictions: dict[str, list[str]], filepath: str, top_n: int = 50):
    submission_data = []

    for qid, items in predictions.items():
        top_n_items = []
        seen = set()
        for item in items:
            if item not in seen:
                top_n_items.append(item)
                seen.add(item)
            if len(top_n_items) == top_n:
                break

        answer_string = " ".join(top_n_items)

        submission_data.append({
            'query_id': qid,
            'answer': answer_string
        })

    df_submission = pd.DataFrame(submission_data)
    df_submission.to_csv(filepath, index=False, encoding='utf-8')