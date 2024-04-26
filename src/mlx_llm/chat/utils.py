STOP = 0
WAIT = 1
GO = 2


def answer_status(text: str, end_str: str) -> int:
    """Check if the model is done generating the end_str

    Args:
        text (str): generated text
        end_str (str): end string

    Returns:
        int: STOP, WAIT, or GO
    """
    if text[-len(end_str) :] == end_str:
        return STOP
    for i in range(len(end_str) - 1, 0, -1):
        if text[-i:] == end_str[:i]:
            return WAIT
    return GO
