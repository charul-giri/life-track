def extract_tasks(text):
    """
    Extracts tasks from the given text. 
    This function looks for tasks in the form of phrases that indicate an action item.
    """
    tasks = []
    # Example of simple task extraction logic
    # You can extend this logic based on your needs.
    phrases = text.split('\n')
    for phrase in phrases:
        if phrase.strip() and phrase.strip()[-1] == ':':
            tasks.append(phrase.strip())
    return tasks
