# test_lifetrack_extraction.py

from task_extractor import extract_tasks_from_paragraph

def run_tests():
    examples = [
        "Today at 8 am I have to prepare the breakfast and at 9 am go to the gym.",
        "Tomorrow evening I will call mom and submit the assignment before 10 pm.",
        "At 7 pm call mom and then go for a walk.",
        "This morning I cleaned the room. Next Monday I will attend the meeting.",
        "On 25th December buy gifts and visit my grandparents in the evening.",
    ]

    for i, text in enumerate(examples, start=1):
        print("=" * 80)
        print(f"Example {i}: {text}")
        tasks = extract_tasks_from_paragraph(text)
        for t in tasks:
            print(f"  TASK: {t['task']!r}\n  TIME: {t['time']!r}")
        print()

if __name__ == "__main__":
    run_tests()
